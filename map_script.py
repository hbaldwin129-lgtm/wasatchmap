import io
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Patch

st.set_page_config(page_title="Wasatch Avalanche Overlay", layout="wide")

# ============================================================
# CONSTANTS
# ============================================================
ALPHA_DANGER = 0.65
ALPHA_SLOPE = 0.20

DANGER_COLORS = {
    "No Rating":    (0.0, 0.0, 0.0, 0.0),
    "Low":          (102/255, 204/255, 0/255, ALPHA_DANGER),    # green
    "Moderate":     (255/255, 204/255, 0/255, ALPHA_DANGER),    # yellow
    "Considerable": (255/255, 153/255, 0/255, ALPHA_DANGER),    # orange
    "High":         (255/255, 0/255, 0/255, ALPHA_DANGER),      # red
    "Extreme":      (0/255, 0/255, 0/255, ALPHA_DANGER),        # black
}

DANGER_RANK = {
    "No Rating": 0,
    "Low": 1,
    "Moderate": 2,
    "Considerable": 3,
    "High": 4,
    "Extreme": 5,
}


# ============================================================
# TERRAIN FUNCTIONS
# ============================================================
def compute_slope_aspect(dem_array, xres, yres):
    """
    Compute slope (degrees) and aspect (degrees clockwise from north)
    using raster gradients.
    """
    dz_dy, dz_dx = np.gradient(dem_array, yres, xres)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect as degrees clockwise from north
    aspect_rad = np.arctan2(dz_dx, -dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 360 + aspect_deg, aspect_deg)

    return slope_deg, aspect_deg


def hillshade(dem, xres, yres, azimuth=315, altitude=45):
    """
    Create a grayscale hillshade for topo-style background.
    """
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)

    dz_dy, dz_dx = np.gradient(dem, yres, xres)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dx, dz_dy)

    shaded = (
        np.sin(altitude_rad) * np.sin(slope)
        + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    )

    shaded = np.nan_to_num(shaded, nan=np.nanmin(shaded))
    if shaded.max() != shaded.min():
        shaded = (shaded - shaded.min()) / (shaded.max() - shaded.min())
    else:
        shaded = np.zeros_like(shaded)

    return shaded


def elevation_band_mask(dem_array, band_name, lower_max, middle_max):
    """
    Convert Lower/Middle/Upper to DEM-based elevation masks.
    """
    band_name = str(band_name).strip().lower()

    if band_name == "lower":
        return dem_array < lower_max
    elif band_name == "middle":
        return (dem_array >= lower_max) & (dem_array < middle_max)
    elif band_name == "upper":
        return dem_array >= middle_max
    else:
        raise ValueError(f"Unknown elevation band: {band_name}")


def aspect_mask(aspect_array, aspect_label):
    """
    Return mask for one of the 8 aspect bins:
    N, NE, E, SE, S, SW, W, NW
    """
    aspect_label = str(aspect_label).strip().upper()

    bins = {
        "N":  [(337.5, 360.0), (0.0, 22.5)],
        "NE": [(22.5, 67.5)],
        "E":  [(67.5, 112.5)],
        "SE": [(112.5, 157.5)],
        "S":  [(157.5, 202.5)],
        "SW": [(202.5, 247.5)],
        "W":  [(247.5, 292.5)],
        "NW": [(292.5, 337.5)],
    }

    if aspect_label not in bins:
        raise ValueError(f"Unknown aspect label: {aspect_label}")

    mask = np.zeros(aspect_array.shape, dtype=bool)

    for lo, hi in bins[aspect_label]:
        if lo < hi:
            mask |= (aspect_array >= lo) & (aspect_array < hi)
        else:
            # wrap-around range
            mask |= (aspect_array >= lo) | (aspect_array < hi)

    return mask


@st.cache_data(show_spinner=False)
def load_excel_data(excel_file):
    danger_df = pd.read_excel(excel_file, sheet_name="DangerRose", engine="openpyxl")
    danger_df.columns = [c.strip() for c in danger_df.columns]

    danger_df["region"] = danger_df["region"].astype(str).str.strip().str.lower()
    danger_df["aspect"] = danger_df["aspect"].astype(str).str.strip().str.upper()
    danger_df["elevation"] = danger_df["elevation"].astype(str).str.strip()
    danger_df["danger_label"] = danger_df["danger_label"].astype(str).str.strip()
    danger_df["forecast_date"] = pd.to_datetime(
        danger_df["forecast_date"]
    ).dt.strftime("%Y-%m-%d")

    return danger_df


@st.cache_data(show_spinner=False)
def read_dem(uploaded_dem_bytes):
    with rasterio.io.MemoryFile(uploaded_dem_bytes) as memfile:
        with memfile.open() as src:
            dem = src.read(1).astype("float32")
            transform = src.transform
            bounds = src.bounds
            xres = abs(transform.a)
            yres = abs(transform.e)
            nodata = src.nodata
            crs = src.crs

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    return {
        "dem": dem,
        "transform": transform,
        "bounds": bounds,
        "extent": extent,
        "xres": xres,
        "yres": yres,
        "crs": str(crs) if crs else None,
    }


def build_map(
    dem,
    extent,
    xres,
    yres,
    region,
    forecast_date,
    danger_today,
    lower_max,
    middle_max,
    slope_min,
    slope_max,
    show_slope_layer,
    show_danger_layer,
):
    # --------------------------------------------------------
    # Terrain derivatives
    # --------------------------------------------------------
    slope_deg, aspect_deg = compute_slope_aspect(dem, xres, yres)
    slope_mask = (slope_deg >= slope_min) & (slope_deg <= slope_max) & np.isfinite(dem)

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------
    dem_filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    shade = hillshade(dem_filled, xres, yres)

    dem_norm = dem_filled.copy()
    dem_min = np.nanmin(dem_norm)
    dem_max = np.nanmax(dem_norm)
    if dem_max > dem_min:
        dem_norm = (dem_norm - dem_min) / (dem_max - dem_min)
    else:
        dem_norm = np.zeros_like(dem_norm)

    # --------------------------------------------------------
    # Avalanche danger layer
    # --------------------------------------------------------
    danger_rank_grid = np.zeros(dem.shape, dtype=np.uint8)
    danger_label_grid = np.full(dem.shape, "No Rating", dtype=object)

    for _, row in danger_today.iterrows():
        elev_class = row["elevation"]
        asp_class = row["aspect"]
        danger_label = row["danger_label"]

        elev_mask = elevation_band_mask(dem, elev_class, lower_max, middle_max)
        asp_mask = aspect_mask(aspect_deg, asp_class)

        combined_mask = slope_mask & elev_mask & asp_mask

        incoming_rank = DANGER_RANK.get(danger_label, 0)
        replace_mask = combined_mask & (incoming_rank > danger_rank_grid)

        danger_rank_grid[replace_mask] = incoming_rank
        danger_label_grid[replace_mask] = danger_label

    danger_overlay = np.zeros((dem.shape[0], dem.shape[1], 4), dtype=np.float32)
    for label, rgba in DANGER_COLORS.items():
        mask = danger_label_grid == label
        danger_overlay[mask, 0] = rgba[0]
        danger_overlay[mask, 1] = rgba[1]
        danger_overlay[mask, 2] = rgba[2]
        danger_overlay[mask, 3] = rgba[3]

    # --------------------------------------------------------
    # Slope mask layer
    # --------------------------------------------------------
    slope_overlay = np.zeros((dem.shape[0], dem.shape[1], 4), dtype=np.float32)
    slope_overlay[slope_mask] = (0.2, 0.5, 1.0, ALPHA_SLOPE)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    # base topo
    ax.imshow(shade, cmap="gray", extent=extent, origin="upper")
    ax.imshow(dem_norm, cmap="terrain", extent=extent, origin="upper", alpha=0.18)

    # optional layers
    if show_slope_layer:
        ax.imshow(slope_overlay, extent=extent, origin="upper")

    if show_danger_layer:
        ax.imshow(danger_overlay, extent=extent, origin="upper")

    ax.set_title(
        f"Wasatch Avalanche Overlay\n"
        f"Region: {region} | Date: {forecast_date}\n"
        f"Danger shown only on slopes between {slope_min}° and {slope_max}°"
    )
    ax.set_xlabel("Map X")
    ax.set_ylabel("Map Y")

    legend_items = []

    if show_danger_layer:
        legend_items.extend([
            Patch(facecolor=DANGER_COLORS["Low"], edgecolor="black", label="Low"),
            Patch(facecolor=DANGER_COLORS["Moderate"], edgecolor="black", label="Moderate"),
            Patch(facecolor=DANGER_COLORS["Considerable"], edgecolor="black", label="Considerable"),
            Patch(facecolor=DANGER_COLORS["High"], edgecolor="black", label="High"),
        ])

    if show_slope_layer:
        legend_items.append(
            Patch(facecolor=(0.2, 0.5, 1.0, ALPHA_SLOPE), edgecolor="black", label=f"{slope_min}–{slope_max}° slope mask")
        )

    if legend_items:
        ax.legend(handles=legend_items, loc="lower left")

    plt.tight_layout()
    return fig, slope_deg, aspect_deg, slope_mask


# ============================================================
# UI
# ============================================================
st.title("Wasatch Avalanche Terrain Overlay")
st.markdown(
    """
Upload:
1. a **DEM GeoTIFF** for the Wasatch  
2. your **avalanche_forecasts.xlsx** workbook

The app will compute:
- slope
- aspect
- 30–45° slope mask (or your chosen range)
- avalanche danger overlay by **aspect** and **elevation band**
"""
)

with st.sidebar:
    st.header("Inputs")

    dem_file = st.file_uploader("Upload DEM (.tif)", type=["tif", "tiff"])
    excel_file = st.file_uploader("Upload avalanche workbook (.xlsx)", type=["xlsx"])

    st.header("Layer Controls")
    show_slope_layer = st.checkbox("Show slope mask layer", value=True)
    show_danger_layer = st.checkbox("Show avalanche danger layer", value=True)

    st.header("Slope Filter")
    slope_min = st.slider("Minimum slope (degrees)", 0, 89, 30)
    slope_max = st.slider("Maximum slope (degrees)", 0, 89, 45)

    st.header("Elevation Band Thresholds")
    lower_max = st.number_input("Lower band max elevation", value=2400)
    middle_max = st.number_input("Middle band max elevation", value=2900)

# ------------------------------------------------------------
# Load workbook to populate selectors
# ------------------------------------------------------------
danger_df = None
if excel_file is not None:
    try:
        danger_df = load_excel_data(excel_file)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")

region = None
forecast_date = None

if danger_df is not None and not danger_df.empty:
    with st.sidebar:
        st.header("Forecast Selection")

        available_regions = sorted(danger_df["region"].dropna().unique().tolist())
        default_region = "salt-lake" if "salt-lake" in available_regions else available_regions[0]
        region = st.selectbox("Region", available_regions, index=available_regions.index(default_region))

        region_dates = sorted(
            danger_df.loc[danger_df["region"] == region, "forecast_date"].dropna().unique().tolist()
        )
        default_date = region_dates[-1]
        forecast_date = st.selectbox("Forecast date", region_dates, index=region_dates.index(default_date))

# ------------------------------------------------------------
# Generate map
# ------------------------------------------------------------
if dem_file is not None and danger_df is not None and region is not None and forecast_date is not None:
    try:
        dem_info = read_dem(dem_file.getvalue())
        dem = dem_info["dem"]
        extent = dem_info["extent"]
        xres = dem_info["xres"]
        yres = dem_info["yres"]

        danger_today = danger_df[
            (danger_df["region"] == region) &
            (danger_df["forecast_date"] == forecast_date)
        ].copy()

        if danger_today.empty:
            st.warning("No DangerRose rows found for the selected region/date.")
        else:
            fig, slope_deg, aspect_deg, slope_mask = build_map(
                dem=dem,
                extent=extent,
                xres=xres,
                yres=yres,
                region=region,
                forecast_date=forecast_date,
                danger_today=danger_today,
                lower_max=lower_max,
                middle_max=middle_max,
                slope_min=slope_min,
                slope_max=slope_max,
                show_slope_layer=show_slope_layer,
                show_danger_layer=show_danger_layer,
            )

            st.pyplot(fig, use_container_width=True)

            # Summary stats
            valid_pixels = np.isfinite(dem).sum()
            slope_pixels = slope_mask.sum()
            pct = (slope_pixels / valid_pixels * 100) if valid_pixels > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Valid DEM pixels", f"{valid_pixels:,}")
            col2.metric("Slope-mask pixels", f"{slope_pixels:,}")
            col3.metric("Percent in slope range", f"{pct:.2f}%")

            # PNG download
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                label="Download map as PNG",
                data=buf,
                file_name=f"wasatch_avalanche_overlay_{region}_{forecast_date}.png",
                mime="image/png",
            )

            with st.expander("Debug / Derived Layers"):
                st.write("**DEM shape:**", dem.shape)
                st.write("**Slope degrees** min/max:", float(np.nanmin(slope_deg)), float(np.nanmax(slope_deg)))
                st.write("**Aspect degrees** min/max:", float(np.nanmin(aspect_deg)), float(np.nanmax(aspect_deg)))
                st.write("**CRS:**", dem_info["crs"])

    except Exception as e:
        st.error(f"Error generating map: {e}")
else:
    st.info("Upload both a DEM GeoTIFF and the avalanche Excel workbook to get started.")
