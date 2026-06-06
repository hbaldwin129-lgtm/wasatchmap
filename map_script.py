import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Custom Backcountry Slope Map", layout="wide")

@st.cache_data
def generate_synthetic_dem(grid_size=100):
    """
    Generates a synthetic terrain model for plug-and-play testing.
    Replace this function with rasterio to load a real GeoTIFF.
    """
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Procedural mountain generation (Elevation in meters)
    Z = 1000 * np.exp(-(X**2 + Y**2)/10) + 500 * np.sin(X*2) * np.cos(Y*2) + 200
    
    # Assume each grid cell is 30x30 meters (standard Landsat/USGS DEM resolution)
    resolution = 30 
    return X, Y, Z, resolution

def calculate_slope(Z, resolution):
    """
    Calculates the slope angle in degrees for each pixel in the elevation matrix.
    """
    # Calculate gradients (rise over run) in X and Y directions
    dy, dx = np.gradient(Z, resolution, resolution)
    
    # Calculate the slope angle in degrees
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg

# --- UI Setup ---
st.title("🏔️ 3D Avalanche Slope Analyzer")
st.markdown("Interactive 3D visualization mapping steepness gradients onto elevation data.")

# Sidebar controls for the "danger zone"
st.sidebar.header("Slope Overlays")
min_slope = st.sidebar.slider("Minimum Danger Angle (°)", 0, 90, 30)
max_slope = st.sidebar.slider("Maximum Danger Angle (°)", 0, 90, 45)

# --- Data Processing ---
X, Y, Z, res = generate_synthetic_dem(grid_size=120)
slope = calculate_slope(Z, res)

# Create a color mask: 
# 1 = Inside the selected slope range (Danger)
# 0 = Outside the range (Safe)
danger_mask = np.where((slope >= min_slope) & (slope <= max_slope), 1, 0)

# --- 3D Rendering with Plotly ---
fig = go.Figure(data=[go.Surface(
    z=Z,
    surfacecolor=danger_mask,
    colorscale=[
        [0, 'rgb(240, 240, 240)'], # White/Grey for safe terrain
        [1, 'rgb(255, 50, 50)']    # Red for danger zone terrain
    ],
    cmin=0,
    cmax=1,
    showscale=False,
    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.1)
)])

fig.update_layout(
    scene=dict(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_title="Elevation (m)",
        aspectratio=dict(x=1, y=1, z=0.4) # Flattens the Z axis slightly for realistic scale
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=700
)

st.plotly_chart(fig, use_container_width=True)