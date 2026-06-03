import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def calculate_masked_averages(file_path):
    # 1. Load the Tertiary Export
    p = Path(file_path)
    data = np.load(p)
    
    print(f"Loaded: {p.name}")
    
    # 2. Extract the grids and masked velocity matrices
    X = data['X']
    Y = data['Y']
    masked_u = data['masked_u']  # shape is likely (n_frames, rows, cols)
    masked_v = data['masked_v']
    
    # 3. Calculate time-averages (ignoring NaNs)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Average across the first dimension (frames/time)
        u_mean = np.nanmean(masked_u, axis=0)
        v_mean = np.nanmean(masked_v, axis=0)
        
    speed_mean = np.hypot(u_mean, v_mean)
    
    print(f"Processed {masked_u.shape[0]} frames.")
    print(f"Mean U range: {np.nanmin(u_mean):.3f} to {np.nanmax(u_mean):.3f}")
    print(f"Mean V range: {np.nanmin(v_mean):.3f} to {np.nanmax(v_mean):.3f}")
    
    # 4. Plot the results
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Downsample for a cleaner quiver plot (adjust 's' if arrows are too dense)
    s = 2 
    q = ax.quiver(X[::s, ::s], Y[::s, ::s], 
                  u_mean[::s, ::s], v_mean[::s, ::s], 
                  speed_mean[::s, ::s], 
                  cmap="viridis", angles="xy", scale_units="xy")
    
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Image coordinates (y points down)
    ax.set_title("Time-Averaged Masked Flow Field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(q, ax=ax, label="Velocity Magnitude")
    
    # Save the plot
    output_png = p.parent / "masked_time_average.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_png}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate masked time averages from a Tertiary Export.")
    parser.add_argument("npz", help="Path to the Tertiary Export .npz")
    args = parser.parse_args()
    
    calculate_masked_averages(args.npz)