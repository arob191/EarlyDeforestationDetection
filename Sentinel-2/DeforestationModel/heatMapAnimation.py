import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rasterio

# Load your heat maps for multiple years
heat_maps = [
    "deforestation_2015_2016.png",
    "deforestation_2017_2018.png",
    "deforestation_2019_2020.png",
    "deforestation_2021_2022.png",
    "deforestation_2023_2024.png"
]

# Create a figure for the animation
fig, ax = plt.subplots(figsize=(10, 6))
img = None

# Function to update the image
def update(frame):
    global img
    if img:
        img.remove()
    heat_map = plt.imread(heat_maps[frame])
    img = ax.imshow(heat_map, animated=True)
    ax.set_title(f"Deforestation Predictions: {2015 + frame*2}-{2016 + frame*2}")
    return img,

# Create animation
ani = FuncAnimation(fig, update, frames=len(heat_maps), interval=1000, blit=True)

# Save as GIF or video
ani.save("deforestation_animation.gif", writer="imagemagick")
plt.show()