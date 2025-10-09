import os
import numpy as np
import matplotlib.pyplot as plt

# === USER SETTINGS ===
width, height = 4096, 4096
depth = 8                  # 8 or 16 bit per pixel
# ======================

# Get all .grey files in current directory
grey_files = [f for f in os.listdir('.') if f.lower().endswith('.grey')]

if not grey_files:
    print("No .grey files found in current directory.")
    exit()

# List files
print("Found .grey files:")
for i, f in enumerate(grey_files, start=1):
    print(f"{i}: {f}")

# Ask user to choose one
choice = input("Enter the number of the image you want to open: ")

try:
    index = int(choice) - 1
    if index < 0 or index >= len(grey_files):
        raise ValueError
except ValueError:
    print("Invalid choice.")
    exit()

filename = grey_files[index]
print(f"\nOpening {filename}...")

# Determine dtype based on bit depth
dtype = np.uint8 if depth == 8 else np.uint16

# Read and reshape image
data = np.fromfile(filename, dtype=dtype)
expected_size = width * height

if data.size != expected_size:
    print(f"Warning: file size ({data.size} pixels) doesn't match {width}x{height} = {expected_size}")
    # Try to guess shape based on file size
    side = int(np.sqrt(data.size))
    print(f"Trying guessed shape {side}x{side}")
    width = height = side

img = data.reshape((height, width))

# Show image
plt.imshow(img, cmap='gray')
plt.title(filename)
plt.axis('off')
plt.show()
