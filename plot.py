import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image # Used to safely open the original image frames
from fileopener import load

def plot_v_original_with_vectors(npz_file_path, img_dir_path=None):
    """
    Reads PIV data, plots baseline stats, and generates a sequence of 
    contour heatmaps overlaid with flow vector arrows and raw image backgrounds.
    """
    print(f"Loading PIV data from: {npz_file_path}...")
    try:
        results = load(npz_file_path)
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return

    n_frames = results.n_pairs
    print(f"Found {n_frames} frames (pairs) with a grid shape of {results.grid_shape}.")
    output_dir = os.path.dirname(npz_file_path)

    # Pre-index/sort image files from your raw folder if it exists
    img_files = []
    if img_dir_path and os.path.exists(img_dir_path):
        print(f"Checking background images directory: {img_dir_path}")
        img_files = sorted([
            f for f in os.listdir(img_dir_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
        ])
        print(f"Found {len(img_files)} original background images.")
    else:
        print("⚠️ Original image directory not provided or not found. Plotting without image background.")

    # ==========================================
    # PLOT 1: Time-Series Line Plot of Mean v (Kept)
    # ==========================================
    mean_v_per_frame = np.nanmean(abs(results.v_original)*results.calv, axis=(1, 2))
    mean_u_per_frame = np.nanmean(abs(results.u_original)*results.calu, axis=(1, 2))
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_frames), mean_v_per_frame, marker='o', markersize=4, linestyle='-', color='tab:orange')
    plt.plot(range(n_frames), mean_u_per_frame, marker='o', markersize=4, linestyle='-', color='tab:blue')
    plt.legend(['Mean |v_original| (m/s)', 'Mean |u_original| (m/s)'])
    plt.title(f"Mean v_original Over Time ({n_frames} frames)")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Vertical Velocity (meters/second)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    trend_plot_path = os.path.join(output_dir, "v_original_trend.png")
    plt.savefig(trend_plot_path, dpi=300)
    print(f"✅ Saved trend plot to: {trend_plot_path}")
    plt.show()

    # ==========================================
    # PLOT 2: 2D Heatmap of the First Frame (Kept)
    # ==========================================
    if n_frames > 0:
        plt.figure(figsize=(8, 6))
        v_frame_0 = results.v_original[0]
        im = plt.imshow(v_frame_0, cmap='coolwarm', origin='upper')
        plt.colorbar(im, label='v_original (px/frame)')
        plt.title("v_original Heatmap (Frame 0)")
        plt.xlabel("Grid Column (X direction)")
        plt.ylabel("Grid Row (Y direction)")
        plt.tight_layout()
        
        heatmap_plot_path = os.path.join(output_dir, "v_original_heatmap_frame0.png")
        plt.savefig(heatmap_plot_path, dpi=300)
        print(f"✅ Saved frame 0 heatmap to: {heatmap_plot_path}")
        plt.show()

    # ==========================================
    # PLOT 3: Contour Maps + Vector Arrows (Frames 50 to 200, step 2)
    # ==========================================
    start_frame = 50
    end_frame = 200
    step = 2
    downsample_arrows = 2 # Change to 1 to show ALL arrows, increase if arrows look too crowded
    
    seq_dir = os.path.join(output_dir, f"v_vector_contours_{start_frame}_to_{end_frame}")
    os.makedirs(seq_dir, exist_ok=True)
    
    print(f"\nGenerating vector contour plots in: {seq_dir}")
    generated_count = 0
    
    for f_idx in range(start_frame, end_frame + 1, step):
        if f_idx >= n_frames:
            print(f" Stopping early: Frame {f_idx} exceeds available PIV frames ({n_frames}).")
            break
            
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # 1. Try loading and plotting the original raw camera image frame as background
        bg_loaded = False
        if f_idx < len(img_files):
            try:
                img_path = os.path.join(img_dir_path, img_files[f_idx])
                bg_img = Image.open(img_path)
                ax.imshow(bg_img, cmap='gray', origin='upper')
                bg_loaded = True
            except Exception as e:
                pass # Fallback smoothly if an image fails to load
        
        # 2. Extract components for this frame
        u_frame = results.u_original[f_idx]
        v_frame = results.v_original[f_idx]
        magnitude_v = np.abs(v_frame)*results.calv # Magnitude of vertical velocity
        
        # 3. Plot the filled contour map (Heatmap)
        # If an image background is loaded, we give it transparency (alpha) so you can see the cells/flow beneath
        alpha_val = 0.55 if bg_loaded else 1.0
        #make the color bar +- .25 m/s, which is a reasonable range for our PIV data (adjust if needed)
        levels = np.linspace(-0.25, 0.25, 30)
        contour = ax.contourf(results.x*results.calu, results.y*results.calv, magnitude_v, levels=levels, cmap='viridis', alpha=alpha_val)
        cbar = fig.colorbar(contour, ax=ax, label='|v_original| Magnitude (m/s)')
        
        # 4. Overlay the Vector Arrows
        s = downsample_arrows
        ax.quiver(
            results.x[::s, ::s], 
            results.y[::s, ::s], 
            u_frame[::s, ::s], 
            v_frame[::s, ::s], 
            color='white',         # White arrows stand out excellently on 'viridis' and gray backgrounds
            edgecolor='black',     # Adds a small dark outline to arrows for crisp visibility
            linewidth=0.5,
            angles='xy', 
            scale_units='xy', 
            scale=1.5,             # Increase scale number to make arrows shorter, decrease to make them longer
            width=0.0035
        )
        
        # Adjust axes configurations
        ax.set_aspect("equal")
        ax.invert_yaxis() # Standard layout coordinate alignment: Image 0,0 is top-left
        ax.set_title(f"Velocity Vector Contour Map (Frame {f_idx})")
        ax.set_xlabel("X coordinate (pixels)")
        ax.set_ylabel("Y coordinate (pixels)")
        fig.tight_layout()
        
        # Save frame and clean memory allocation
        seq_plot_path = os.path.join(seq_dir, f"v_contour_vec_frame_{f_idx:03d}.png")
        plt.savefig(seq_plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        generated_count += 1
        
    print(f"✅ Successfully generated {generated_count} vector contour plots!")

if __name__ == "__main__":
    # File Paths
    piv_data_path = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data\PipelineTest_new\Dynamic Masks\dynamic_masking_firstsave.npz"
    piv_data_path = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data\Test1\firstsave.npz" 
    #piv_data_path = r"C:\Users\JackHu\Downloads\primary_export.npz"
    #piv_data_path_mat =r"C:\Users\JackHu\Downloads\synthetic_primary_export.mat"
    raw_images_path = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data\Test1\100mL-20deg-35cpm"
    
    # Execution
    plot_v_original_with_vectors(piv_data_path, raw_images_path)