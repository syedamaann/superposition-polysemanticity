import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
n_neurons = 49  # Reduced number of neurons for clearer visualization (7x7 grid)
n_objects = 3   # Number of different objects to represent
n_frames = 180  # Number of frames in the animation
neuron_grid_size = int(np.sqrt(n_neurons))  # Grid size for visualization

# Create custom colormaps for neuron activation
colors_main = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0)]
cmap_main = LinearSegmentedColormap.from_list('neuron_activation', colors_main, N=100)

# Colors for each object
object_colors = [
    [(0, 0, 0), (1, 0, 0)],  # Red for first object
    [(0, 0, 0), (0, 1, 0)],  # Green for second object
    [(0, 0, 0), (0, 0, 1)]   # Blue for third object
]
cmaps = [LinearSegmentedColormap.from_list(f'object_{i}', colors, N=100) for i, colors in enumerate(object_colors)]

# Object names for our visualization
object_names = ["Cat", "Car", "House"]

# Generate random neuron activations for different objects
# In superposition, the same neurons can represent multiple objects
base_activations = np.zeros((n_objects, n_neurons))
shared_neuron_maps = np.zeros((n_objects, n_objects, n_neurons), dtype=bool)

# Create distinct but overlapping activation patterns for each object
for i in range(n_objects):
    # Each object activates a subset of neurons
    primary_neurons = np.random.choice(n_neurons, size=int(n_neurons * 0.3), replace=False)
    base_activations[i, primary_neurons] = np.random.uniform(0.6, 1.0, size=len(primary_neurons))
    
    # Create overlapping neurons with other objects
    for j in range(n_objects):
        if i != j:
            # Select some neurons to be shared with object j
            shared_count = int(len(primary_neurons) * 0.4)
            shared_neurons = np.random.choice(primary_neurons, size=shared_count, replace=False)
            
            # Mark these neurons as shared between objects i and j
            shared_neuron_maps[i, j, shared_neurons] = True
            shared_neuron_maps[j, i, shared_neurons] = True
            
            # Ensure object j also activates these shared neurons
            if j < i:  # Only adjust for objects we've already processed
                base_activations[j, shared_neurons] = np.clip(
                    base_activations[j, shared_neurons] + np.random.uniform(0.2, 0.5, size=shared_count),
                    0, 1
                )

# Function to create a frame of neuron activations
def create_activation_frame(base_activation, progress):
    """Create a frame of neuron activations with a wave-like activation pattern."""
    # Create a wave-like activation pattern
    activation = base_activation * progress
    
    # Apply Gaussian filter for a smoother, more natural looking activation
    activation = gaussian_filter(activation, sigma=0.5)
    
    return activation.reshape(neuron_grid_size, neuron_grid_size)

# Set up the figure for animation
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Enhanced Visualization of Neural Network Superposition", fontsize=22, y=0.98)

# Create a grid layout
grid = plt.GridSpec(2, 4, height_ratios=[1, 1.2], width_ratios=[1, 1, 1, 0.1], hspace=0.3)

# Top row: Individual object activations
axes_objects = []
images_objects = []
for i in range(n_objects):
    ax = fig.add_subplot(grid[0, i])
    ax.set_title(f"Object: {object_names[i]}", fontsize=16)
    img = ax.imshow(np.zeros((neuron_grid_size, neuron_grid_size)), 
                   cmap=cmaps[i], vmin=0, vmax=1, animated=True)
    images_objects.append(img)
    ax.set_xticks([])
    ax.set_yticks([])
    axes_objects.append(ax)
    
    # Add grid lines to make neurons more distinct
    ax.grid(True, color='white', linestyle='-', linewidth=0.5)

# Add a colorbar for the top row
cax_top = fig.add_subplot(grid[0, 3])
cbar_top = fig.colorbar(images_objects[0], cax=cax_top)
cbar_top.set_label('Neuron Activation Level', fontsize=12)

# Bottom row: Combined activation and superposition visualization
ax_combined = fig.add_subplot(grid[1, :3])
ax_combined.set_title("Combined Neural Activations (Superposition View)", fontsize=18)
img_combined = ax_combined.imshow(np.zeros((neuron_grid_size, neuron_grid_size)), 
                                 cmap='viridis', vmin=0, vmax=1, animated=True)
ax_combined.set_xticks([])
ax_combined.set_yticks([])

# Add grid lines to make neurons more distinct
ax_combined.grid(True, color='white', linestyle='-', linewidth=0.5)

# Create a legend for the combined view
legend_elements = [
    patches.Patch(facecolor='red', edgecolor='black', alpha=0.7, label=f'{object_names[0]} neurons'),
    patches.Patch(facecolor='green', edgecolor='black', alpha=0.7, label=f'{object_names[1]} neurons'),
    patches.Patch(facecolor='blue', edgecolor='black', alpha=0.7, label=f'{object_names[2]} neurons'),
    patches.Patch(facecolor='purple', edgecolor='black', alpha=0.7, label='Shared neurons (Superposition)')
]
ax_combined.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Animation update function
def update(frame):
    # Calculate which phase we're in
    cycle = frame // (n_frames // 3)
    phase_frame = frame % (n_frames // 3)
    progress = phase_frame / (n_frames // 3)
    
    # Determine which object is currently active
    current_obj = cycle % n_objects
    
    # Update individual object visualizations
    for i in range(n_objects):
        if i == current_obj:
            # Current object is activating
            activation = create_activation_frame(base_activations[i], progress)
            images_objects[i].set_array(activation)
        else:
            # Show residual activation for other objects
            decay = max(0, 1 - (phase_frame / 10))
            if cycle > 0 and shared_neuron_maps[current_obj, i].any():
                # Show shared neurons with current object
                shared_activation = np.zeros_like(base_activations[i])
                shared_activation[shared_neuron_maps[current_obj, i]] = base_activations[i][shared_neuron_maps[current_obj, i]] * progress * 0.7
                images_objects[i].set_array(create_activation_frame(shared_activation, 1.0))
            else:
                images_objects[i].set_array(np.zeros((neuron_grid_size, neuron_grid_size)))
    
    # Update combined visualization
    combined_activation = np.zeros((neuron_grid_size, neuron_grid_size, 3))
    
    # Add current object's activation to the RGB channels
    obj_activation = create_activation_frame(base_activations[current_obj], progress)
    
    # Red channel for object 0
    if current_obj == 0:
        combined_activation[:,:,0] = obj_activation
    elif shared_neuron_maps[current_obj, 0].any():
        shared = shared_neuron_maps[current_obj, 0].reshape(neuron_grid_size, neuron_grid_size)
        combined_activation[:,:,0][shared] = obj_activation[shared] * 0.7
    
    # Green channel for object 1
    if current_obj == 1:
        combined_activation[:,:,1] = obj_activation
    elif shared_neuron_maps[current_obj, 1].any():
        shared = shared_neuron_maps[current_obj, 1].reshape(neuron_grid_size, neuron_grid_size)
        combined_activation[:,:,1][shared] = obj_activation[shared] * 0.7
    
    # Blue channel for object 2
    if current_obj == 2:
        combined_activation[:,:,2] = obj_activation
    elif shared_neuron_maps[current_obj, 2].any():
        shared = shared_neuron_maps[current_obj, 2].reshape(neuron_grid_size, neuron_grid_size)
        combined_activation[:,:,2][shared] = obj_activation[shared] * 0.7
    
    # Highlight shared neurons in the combined view
    for i in range(n_objects):
        if i != current_obj:
            shared = shared_neuron_maps[current_obj, i].reshape(neuron_grid_size, neuron_grid_size)
            # Make shared neurons appear as a mix of colors (purplish)
            if i == 0:  # Red
                combined_activation[:,:,0][shared] = obj_activation[shared] * progress
            elif i == 1:  # Green
                combined_activation[:,:,1][shared] = obj_activation[shared] * progress
            else:  # Blue
                combined_activation[:,:,2][shared] = obj_activation[shared] * progress
    
    img_combined.set_array(np.clip(combined_activation, 0, 1))
    
    return images_objects + [img_combined]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=100)

# Adjust layout
plt.tight_layout(rect=[0, 0.08, 1, 0.9])

# Save the animation
ani.save('enhanced_superposition.mp4', writer='ffmpeg', fps=10, dpi=150)

# Display the animation (will only work in interactive environments)
plt.show() 