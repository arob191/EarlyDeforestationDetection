import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image

# List of PNG heat maps for each year
heat_maps = [
    "C:\\Users\\Austin\\OneDrive\\Documents\\Personal Projects\\GitHub\\EarlyDeforestationDetection\\deforestation_heatmap_year_1.png",
    "C:\\Users\\Austin\\OneDrive\\Documents\\Personal Projects\\GitHub\\EarlyDeforestationDetection\\deforestation_heatmap_year_2.png",
    "C:\\Users\\Austin\\OneDrive\\Documents\\Personal Projects\\GitHub\\EarlyDeforestationDetection\\deforestation_heatmap_year_3.png",
    "C:\\Users\\Austin\\OneDrive\\Documents\\Personal Projects\\GitHub\\EarlyDeforestationDetection\\deforestation_heatmap_year_4.png",
    "C:\\Users\\Austin\\OneDrive\\Documents\\Personal Projects\\GitHub\\EarlyDeforestationDetection\\deforestation_heatmap_year_5.png"
]

# Compute cumulative differences relative to Year 1
def compute_cumulative_differences(heat_maps):
    differences = []
    # Load Year 1 heat map as baseline
    base_heat_map = np.array(Image.open(heat_maps[0]).convert("L"), dtype=np.float32)
    for i in range(1, len(heat_maps)):
        # Load current year's heat map
        current_heat_map = np.array(Image.open(heat_maps[i]).convert("L"), dtype=np.float32)
        # Compute cumulative difference
        diff = current_heat_map - base_heat_map
        differences.append(diff)
    return differences

# Compute differences relative to Year 1
cumulative_differences = compute_cumulative_differences(heat_maps)

# Create a figure for the animation
fig, ax = plt.subplots(figsize=(10, 6))
img = None

# Function to update the animation
def update(frame):
    global img
    if img:
        img.remove()
    difference = cumulative_differences[frame]
    img = ax.imshow(difference, cmap="coolwarm", animated=True, vmin=-np.max(np.abs(difference)), vmax=np.max(np.abs(difference)))
    ax.set_title(f"Cumulative Change in Deforestation: Year 1 to Year {frame + 2}")
    return img,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(cumulative_differences), interval=1000, blit=True)

# Save the animation as a GIF
ani.save("cumulative_deforestation_animation.gif", writer="imagemagick")
plt.show()