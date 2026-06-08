import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import CheckButtons

# ============================================================
# USER SETTINGS
# ============================================================
DEM_PATH = "wasatch_dem.tif"
EXCEL_PATH = "avalanche_forecasts.xlsx"
REGION = "salt-lake"
FORECAST_DATE = "2026-02-21"
OUTPUT_PNG = "wasatch_avalanche_overlay.png"

# Elevation-band thresholds for classifying Lower / Middle / Upper
# Adjust these to match your preferred Wasatch cutoffs.
# Assumes DEM is in METERS.
LOWER_MAX = 2400
MIDDLE_MAX = 2900

# Visual settings
ALPHA_DANGER = 0.65
ALPHA_SLOPE = 0.20

# ============================================================
# DANGER COLORS
# ============================================================
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

def elevation_band_mask(dem_array, band_name):
    """
    Convert UAC band labels to DEM-based elevation masks.
    """
    band_name = str(band_name).strip().lower()

    if band_name == "lower":
        return dem_array < LOWER_MAX
    elif band_name == "middle":
        return (dem_array >= LOWER_MAX) & (dem_array < MIDDLE_MAX)
    elif band_name == "upper":
        return dem_array >= MIDDLE_MAX
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
            # wraparound sector
            mask |= (aspect_array >= lo) | (aspect_array < hi)

    return mask

# ============================================================
# LOAD DEM
# ============================================================
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype("float32")
    transform = src.transform
    bounds = src.bounds
    xres = abs(transform.a)
    yres = abs(transform.e)
    nodata = src.nodata

if nodata is not None:
    dem = np.where(dem == nodata, np.nan, dem)

# Useful map extent for imshow
extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# ============================================================
# COMPUTE TERRAIN DERIVATIVES
# ============================================================
slope_deg, aspect_deg = compute_slope_aspect(dem, xres, yres)

# First layer: slopes between 30 and 45 degrees
slope_mask = (slope_deg >= 30) & (slope_deg <= 45) & np.isfinite(dem)

# ============================================================
# BUILD TOPO BACKGROUND
# ============================================================
dem_filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
shade = hillshade(dem_filled, xres, yres)

# Optional faint elevation tint for topo feel
dem_norm = dem_filled.copy()
valid = np.isfinite(dem_norm)
dem_min = np.nanmin(dem_norm)
dem_max = np.nanmax(dem_norm)
if dem_max > dem_min:
    dem_norm = (dem_norm - dem_min) / (dem_max - dem_min)
else:
    dem_norm = np.zeros_like(dem_norm)

# ============================================================
# READ AVALANCHE DATA
# ============================================================
danger_df = pd.read_excel(EXCEL_PATH, sheet_name="DangerRose", engine="openpyxl")
danger_df.columns = [c.strip() for c in danger_df.columns]

danger_df["region"] = danger_df["region"].astype(str).str.strip().str.lower()
danger_df["aspect"] = danger_df["aspect"].astype(str).str.strip().str.upper()
danger_df["elevation"] = danger_df["elevation"].astype(str).str.strip()
danger_df["danger_label"] = danger_df["danger_label"].astype(str).str.strip()
danger_df["forecast_date"] = pd.to_datetime(danger_df["forecast_date"]).dt.strftime("%Y-%m-%d")

danger_today = danger_df[
    (danger_df["region"] == REGION.lower()) &
    (danger_df["forecast_date"] == FORECAST_DATE)
].copy()

if danger_today.empty:
    raise ValueError(
        f"No DangerRose rows found for region='{REGION}' and date='{FORECAST_DATE}'."
    )

# ============================================================
# BUILD AVALANCHE DANGER GRID
# ============================================================
danger_rank_grid = np.zeros(dem.shape, dtype=np.uint8)

for _, row in danger_today.iterrows():
    elev_class = row["elevation"]
    asp_class = row["aspect"]
    danger_label = row["danger_label"]

    elev_mask = elevation_band_mask(dem, elev_class)
    asp_mask = aspect_mask(aspect_deg, asp_class)

    combined_mask = slope_mask & elev_mask & asp_mask

    incoming_rank = DANGER_RANK.get(danger_label, 0)
    replace_mask = combined_mask & (incoming_rank > danger_rank_grid)

    danger_rank_grid[replace_mask] = incoming_rank

# RGBA danger overlay
danger_overlay = np.zeros((dem.shape[0], dem.shape[1], 4), dtype=np.float32)

for label, rgba in DANGER_COLORS.items():
    rank = DANGER_RANK.get(label, 0)
    mask = danger_rank_grid == rank
    danger_overlay[mask, 0] = rgba[0]
    danger_overlay[mask, 1] = rgba[1]
    danger_overlay[mask, 2] = rgba[2]
    danger_overlay[mask, 3] = rgba[3]

# Slope-only overlay (removable/toggleable)
slope_overlay = np.zeros((dem.shape[0], dem.shape[1], 4), dtype=np.float32)
slope_overlay[slope_mask] = (0.2, 0.5, 1.0, ALPHA_SLOPE)  # blue transparent

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(12, 12))
plt.subplots_adjust(left=0.08, right=0.83)

# 1) topo background
bg = ax.imshow(shade, cmap="gray", extent=extent, origin="upper")

# 2) optional subtle elevation tint for more topo feel
elev_tint = ax.imshow(dem_norm, cmap="terrain", extent=extent, origin="upper", alpha=0.18)

# 3) removable slope layer
slope_img = ax.imshow(slope_overlay, extent=extent, origin="upper", visible=True)

# 4) avalanche danger layer
danger_img = ax.imshow(danger_overlay, extent=extent, origin="upper", visible=True)

ax.set_title(
    f"Wasatch Avalanche Overlay\n"
    f"Region: {REGION} | Date: {FORECAST_DATE}\n"
    f"Danger shown only on slopes between 30° and 45°"
)
ax.set_xlabel("Map X")
ax.set_ylabel("Map Y")

legend_items = [
    Patch(facecolor=DANGER_COLORS["Low"], edgecolor="black", label="Low"),
    Patch(facecolor=DANGER_COLORS["Moderate"], edgecolor="black", label="Moderate"),
    Patch(facecolor=DANGER_COLORS["Considerable"], edgecolor="black", label="Considerable"),
    Patch(facecolor=DANGER_COLORS["High"], edgecolor="black", label="High"),
    Patch(facecolor=(0.2, 0.5, 1.0, ALPHA_SLOPE), edgecolor="black", label="30–45° slope mask"),
]

ax.legend(handles=legend_items, loc="lower left")

# ============================================================
# REMOVABLE LAYERS VIA CHECKBOXES
# ============================================================
rax = plt.axes([0.85, 0.45, 0.13, 0.12])
labels = ["Slope mask", "Danger layer"]
visibility = [True, True]
check = CheckButtons(rax, labels, visibility)

def toggle_layer(label):
    if label == "Slope mask":
        slope_img.set_visible(not slope_img.get_visible())
    elif label == "Danger layer":
        danger_img.set_visible(not danger_img.get_visible())
    plt.draw()

check.on_clicked(toggle_layer)

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved map to: {OUTPUT_PNG}")

plt.show()
``
