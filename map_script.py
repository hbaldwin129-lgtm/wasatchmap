import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from PIL import Image
import math
from io import BytesIO

st.set_page_config(page_title="Wasatch Slope Map", layout="wide")
st.title("🏔️ Wasatch 3D Avalanche Slope Map")
st.write("Loading real Wasatch elevation data piece by piece.")

# ------------------------------------------------------------
# 1. SLIPPY MAP MATH
# ------------------------------------------------------------

def latlon_to_tile(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom

    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    return xtile, ytile


def get_tile_url(x, y, zoom):
    return f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png"


def decode_terrarium_tile(img):
    img_array = np.array(img)

    R = img_array[:, :, 0].astype(np.float32)
    G = img_array[:, :, 1].astype(np.float32)
    B = img_array[:, :, 2].astype(np.float32)

    elevation = (R * 256.0 + G + B / 256.0) - 32768.0
    return elevation


@st.cache_data(show_spinner=False)
def download_tile(x, y, zoom):
    url = get_tile_url(x, y, zoom)

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")
    elevation = decode_terrarium_tile(img)

    return elevation


# ------------------------------------------------------------
# 2. CREATE TILE LIST FOR A BOUNDING BOX
# ------------------------------------------------------------

def get_tile_grid(north, south, west, east, zoom):
    x_min, y_max = latlon_to_tile(south, west, zoom)
    x_max, y_min = latlon_to_tile(north, east, zoom)

    tile_list = []

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            tile_list.append((x, y))

    return tile_list, x_min, x_max, y_min, y_max


def stitch_loaded_tiles(loaded_tiles, x_min, x_max, y_min, y_max):
    """
    Creates a DEM mosaic from loaded tiles.
    Missing tiles are filled with NaN so the map can render partially.
    """

    rows = []

    for y in range(y_min, y_max + 1):
        row_tiles = []

        for x in range(x_min, x_max + 1):
            key = f"{x}_{y}"

            if key in loaded_tiles:
                row_tiles.append(loaded_tiles[key])
            else:
                row_tiles.append(np.full((256, 256), np.nan, dtype=np.float32))

        rows.append(np.hstack(row_tiles))

    Z = np.vstack(rows)
    return Z


# ------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# ------------------------------------------------------------

st.sidebar.header("Map Area")

area_choice = st.sidebar.selectbox(
    "Choose Region",
    [
        "Wasatch Front",
        "Central Wasatch",
        "Mount Timpanogos",
        "Little Cottonwood / Big Cottonwood",
        "Custom"
    ]
)

if area_choice == "Wasatch Front":
    north, south, west, east = 41.35, 39.95, -112.15, -111.35
    zoom = 10

elif area_choice == "Central Wasatch":
    north, south, west, east = 40.85, 40.45, -111.95, -111.45
    zoom = 11

elif area_choice == "Mount Timpanogos":
    north, south, west, east = 40.50, 40.25, -111.80, -111.45
    zoom = 12

elif area_choice == "Little Cottonwood / Big Cottonwood":
    north, south, west, east = 40.72, 40.50, -111.90, -111.55
    zoom = 12

else:
    north = st.sidebar.number_input("North Latitude", value=41.35)
    south = st.sidebar.number_input("South Latitude", value=39.95)
    west = st.sidebar.number_input("West Longitude", value=-112.15)
    east = st.sidebar.number_input("East Longitude", value=-111.35)
    zoom = st.sidebar.slider("Zoom", 9, 13, 10)

st.sidebar.header("Tile Loading")

tiles_per_click = st.sidebar.slider(
    "Tiles to load per click",
    min_value=1,
    max_value=20,
    value=6,
    step=1
)

downsample = st.sidebar.slider(
    "Downsample factor",
    min_value=1,
    max_value=8,
    value=2,
    step=1,
    help="Higher values render faster but reduce visual detail."
)

st.sidebar.header("Slope Controls")

danger_min = st.sidebar.slider("Minimum Angle", 0, 60, 30)
danger_max = st.sidebar.slider("Maximum Angle", 0, 90, 45)

st.sidebar.header("Topo Line Controls")

show_topo_lines = st.sidebar.checkbox("Show Elevation Topo Lines", value=True)

contour_interval = st.sidebar.slider(
    "Elevation Contour Interval, meters",
    min_value=20,
    max_value=250,
    value=50,
    step=10
)


# ------------------------------------------------------------
# 4. INITIALIZE SESSION STATE
# ------------------------------------------------------------

tile_list, x_min, x_max, y_min, y_max = get_tile_grid(
    north=north,
    south=south,
    west=west,
    east=east,
    zoom=zoom
)

map_key = f"{area_choice}_{north}_{south}_{west}_{east}_{zoom}"

if "map_key" not in st.session_state:
    st.session_state.map_key = map_key

if st.session_state.map_key != map_key:
    st.session_state.map_key = map_key
    st.session_state.loaded_tiles = {}
    st.session_state.next_tile_index = 0

if "loaded_tiles" not in st.session_state:
    st.session_state.loaded_tiles = {}

if "next_tile_index" not in st.session_state:
    st.session_state.next_tile_index = 0


# ------------------------------------------------------------
# 5. LOAD TILES PIECE BY PIECE
# ------------------------------------------------------------

total_tiles = len(tile_list)
loaded_count = len(st.session_state.loaded_tiles)

st.info(f"Loaded {loaded_count} of {total_tiles} tiles.")

progress_value = 0 if total_tiles == 0 else loaded_count / total_tiles
st.progress(progress_value)

col1, col2, col3 = st.columns(3)

with col1:
    load_more = st.button("Load More Tiles")

with col2:
    load_all = st.button("Load All Remaining")

with col3:
    reset_tiles = st.button("Reset Loaded Tiles")

if reset_tiles:
    st.session_state.loaded_tiles = {}
    st.session_state.next_tile_index = 0
    st.rerun()

if load_more or load_all:
    if load_all:
        number_to_load = total_tiles - st.session_state.next_tile_index
    else:
        number_to_load = tiles_per_click

    start_index = st.session_state.next_tile_index
    end_index = min(start_index + number_to_load, total_tiles)

    with st.spinner(f"Downloading tiles {start_index + 1} to {end_index} of {total_tiles}..."):
        for i in range(start_index, end_index):
            x, y = tile_list[i]
            key = f"{x}_{y}"

            if key not in st.session_state.loaded_tiles:
                try:
                    elevation_tile = download_tile(x, y, zoom)
                    st.session_state.loaded_tiles[key] = elevation_tile
                except Exception as e:
                    st.warning(f"Could not load tile {x}, {y}: {e}")

        st.session_state.next_tile_index = end_index

    st.rerun()


# ------------------------------------------------------------
# 6. RENDER ONLY IF SOMETHING HAS LOADED
# ------------------------------------------------------------

if len(st.session_state.loaded_tiles) == 0:
    st.warning("No tiles loaded yet. Click **Load More Tiles** to begin.")
    st.stop()

Z = stitch_loaded_tiles(
    st.session_state.loaded_tiles,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max
)

# Downsample for faster rendering.
Z = Z[::downsample, ::downsample]

# Approx meters per pixel.
center_lat = (north + south) / 2
base_meters_per_pixel = 156543.03392 / (2 ** zoom)

dx = base_meters_per_pixel * math.cos(math.radians(center_lat)) * downsample
dy = base_meters_per_pixel * downsample

x_dist = np.arange(Z.shape[1]) * dx
y_dist = np.arange(Z.shape[0]) * dy
X, Y = np.meshgrid(x_dist, y_dist)

# Fill NaNs temporarily for gradient calculation.
# Plotly will still skip NaN areas in the surface.
Z_for_gradient = np.copy(Z)
nan_mask = np.isnan(Z_for_gradient)

if np.any(~nan_mask):
    Z_for_gradient[nan_mask] = np.nanmean(Z_for_gradient)
else:
    st.warning("Loaded tiles contain no valid elevation data.")
    st.stop()

dz_dy, dz_dx = np.gradient(Z_for_gradient, dy, dx)

slope_deg = np.degrees(
    np.arctan(
        np.sqrt(dz_dx**2 + dz_dy**2)
    )
)

slope_deg[nan_mask] = np.nan

# ------------------------------------------------------------
# 7. COLOR MAPPING
# ------------------------------------------------------------

norm_min = danger_min / 90.0
norm_max = danger_max / 90.0

slope_colorscale = [
    [0.0, "rgb(230, 230, 230)"],
    [norm_min, "rgb(230, 230, 230)"],
    [norm_min, "rgb(255, 255, 0)"],
    [(norm_min + norm_max) / 2, "rgb(255, 165, 0)"],
    [norm_max, "rgb(255, 0, 0)"],
    [norm_max, "rgb(150, 150, 150)"],
    [1.0, "rgb(100, 100, 100)"]
]

valid_z = Z[~np.isnan(Z)]

if len(valid_z) > 0:
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
else:
    z_min = 0
    z_max = 1000

contours_z = dict(
    show=show_topo_lines,
    start=math.floor(z_min / contour_interval) * contour_interval,
    end=math.ceil(z_max / contour_interval) * contour_interval,
    size=contour_interval,
    color="black",
    width=2,
    usecolormap=False,
    highlight=False,
    project=dict(z=False)
)

# ------------------------------------------------------------
# 8. PLOT
# ------------------------------------------------------------

fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    surfacecolor=slope_deg,
    colorscale=slope_colorscale,
    cmin=0,
    cmax=90,
    contours=dict(
        z=contours_z
    ),
    colorbar=dict(
        title="Slope (°)",
        ticksuffix="°"
    )
)])

fig.update_layout(
    scene=dict(
        aspectmode="data",
        xaxis_title="East-West Distance, meters",
        yaxis_title="North-South Distance, meters",
        zaxis_title="Elevation, meters"
    ),
    height=800,
    margin=dict(l=0, r=0, b=0, t=0)
)

st.plotly_chart(fig, use_container_width=True)
