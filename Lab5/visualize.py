import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find and sort all frame files in the 'frames' directory
frames_dir = os.path.join(script_dir, "frames")
if not os.path.exists(frames_dir):
    print(f"Directory '{frames_dir}' not found!")
    exit()

files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.csv")))

if not files:
    print("No frame_*.csv files found!")
    exit()

print(f"Loading {len(files)} frames...")
frames_data = []
for f in files:
    df = pd.read_csv(f)
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        continue
    frames_data.append(df)

if not frames_data:
    print("No valid data frames found!")
    exit()

# Calculate tighter global limits for the data
all_x = pd.concat([df['x'] for df in frames_data])
all_y = pd.concat([df['y'] for df in frames_data])
all_z = pd.concat([df['z'] for df in frames_data])

# Use a percentile to ignore potential extreme outliers for a tighter fit
x_min, x_max = all_x.quantile(0.01), all_x.quantile(0.99)
y_min, y_max = all_y.quantile(0.01), all_y.quantile(0.99)
z_min, z_max = all_z.quantile(0.01), all_z.quantile(0.99)

x_mid = (x_max + x_min) / 2
y_mid = (y_max + y_min) / 2
z_mid = (z_max + z_min) / 2

# Calculate a tighter limit (85% of the range to zoom in slightly)
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
limit = (max_range / 2) * 0.8 

# Create high-resolution figure (3000x3000px)
fig = plt.figure(figsize=(10, 10), facecolor='black', dpi=300)
ax = fig.add_subplot(111, projection='3d', facecolor='black')
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

os.makedirs("plots", exist_ok=True)

def draw_frame(frame_idx):
    ax.clear()
    df = frames_data[frame_idx]
    
    # Refined stars for high resolution (smaller and sharper)
    ax.scatter(df['x'], df['y'], df['z'], s=1.0, c='wheat', alpha=0.9, edgecolors='none')
    
    ax.set_xlim(x_mid - limit, x_mid + limit)
    ax.set_ylim(y_mid - limit, y_mid + limit)
    ax.set_zlim(z_mid - limit, z_mid + limit)
    
    ax.set_axis_off()
    
    # Optional: add a small iteration counter in white
    ax.text2D(0.05, 0.95, f"Iteration {frame_idx}", transform=ax.transAxes, color='white', fontsize=8)
    
    fig.canvas.draw_idle()
    fig.savefig(f"plots/plot_{frame_idx:04d}.png", facecolor='black', edgecolor='none', pad_inches=0)
    print(f"Exported frame {frame_idx}/{len(files)-1}", end='\r')

class AutoPlayer:
    def __init__(self):
        self.ind = 0
        self.timer = fig.canvas.new_timer(interval=50) # Fast export
        self.timer.add_callback(self.step)
        self.timer.start()

    def step(self):
        if self.ind < len(files):
            draw_frame(self.ind)
            self.ind += 1
        else:
            self.timer.stop()
            print("\nExport Complete! You can now run the ffmpeg command.")
            # plt.close(fig) # Uncomment if you want it to close automatically

player = AutoPlayer()
plt.show()
