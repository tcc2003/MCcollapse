import glob
import h5py
import matplotlib.pyplot as plt
import imageio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 

file_list = sorted(glob.glob("./output/*.hdf5"))
output_filename = "movie_3d.mp4"


with h5py.File(file_list[0], 'r') as f:
    coords = f['PartType0/Coordinates'][:]

    x_min, x_max = coords[:, 0].min() - 0.1, coords[:, 0].max() + 0.1
    y_min, y_max = coords[:, 1].min() - 0.1, coords[:, 1].max() + 0.1
    z_min, z_max = coords[:, 2].min() - 0.1, coords[:, 2].max() + 0.1


with imageio.get_writer(output_filename, fps=10) as writer:
    
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    for idx, file_path in enumerate(file_list):
        ax.clear() 
        
        with h5py.File(file_path, 'r') as f:
            coords = f['PartType0/Coordinates'][:]
            rho = f['PartType0/Density'][:]
        
        step = 10  
        x = coords[::step, 0]
        y = coords[::step, 1]
        z = coords[::step, 2]
        
        scat = ax.scatter(x, y, z, s=0.05, alpha=0.3, c=rho[::step], cmap='viridis')
        
        ax.set_box_aspect([1, 1, 1]) 

        ax.set_xlim(0.4 ,0.6)
        ax.set_ylim(0.4, 0.6)
        ax.set_zlim(0.4, 0.6)
        
        ax.view_init(elev=25, azim=45)

        year = idx * 1000        
        ax.set_title(f"{year:,} yr")
    
        ax.set_xlabel("X (pc)")
        ax.set_ylabel("Y (pc)")
        ax.set_zlabel("Z (pc)")
        
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        writer.append_data(frame)
        
    plt.close(fig)

print("Complete!")
