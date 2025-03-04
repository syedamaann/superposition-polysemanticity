import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
n_neurons = 15  # Fewer neurons for clearer visualization
n_objects = 3   # Number of different objects to represent
n_frames = 240  # Number of frames in the animation

# Object names for our visualization
object_names = ["Cat", "Car", "House"]
object_colors = ['red', 'green', 'blue']

# Generate random neuron activations for different objects
# For polysemanticity, we want to show how individual neurons contribute to multiple objects
base_activations = np.zeros((n_objects, n_neurons))

# Create activation patterns for each object
for i in range(n_objects):
    # Each object activates a subset of neurons with varying strengths
    for j in range(n_neurons):
        # Randomly determine if this neuron contributes to this object
        if np.random.random() < 0.6:  # 60% chance of activation
            base_activations[i, j] = np.random.uniform(0.3, 1.0)

# Ensure some neurons are strongly polysemantic (contribute to all objects)
polysemantic_neurons = np.random.choice(n_neurons, size=int(n_neurons * 0.2), replace=False)
for neuron in polysemantic_neurons:
    for i in range(n_objects):
        base_activations[i, neuron] = np.random.uniform(0.7, 1.0)

# Set up the figure for animation
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Neural Network Polysemanticity Visualization", fontsize=24, y=0.98)

# Create a grid layout
gs = GridSpec(3, 4, height_ratios=[1, 1, 1], hspace=0.4)

# Top row: Individual object activations
axes_objects = []
bar_containers = []

for i in range(n_objects):
    ax = fig.add_subplot(gs[i, :3])
    ax.set_title(f"Object: {object_names[i]}", fontsize=16)
    ax.set_xlim(-0.5, n_neurons - 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Neuron ID", fontsize=12)
    ax.set_ylabel("Activation", fontsize=12)
    
    # Create bar plot for neuron activations
    bars = ax.bar(
        range(n_neurons), 
        np.zeros(n_neurons), 
        color=object_colors[i], 
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    bar_containers.append(bars)
    axes_objects.append(ax)
    
    # Add neuron labels
    ax.set_xticks(range(n_neurons))
    ax.set_xticklabels([f"N{i}" for i in range(n_neurons)], fontsize=10, rotation=45)

# Bottom row: Polysemanticity visualization
ax_poly = fig.add_subplot(gs[:, 3])
ax_poly.set_title("Neuron Polysemanticity\n(Contribution to Objects)", fontsize=16)
ax_poly.set_xlim(-0.5, n_objects - 0.5)
ax_poly.set_ylim(-0.5, n_neurons - 0.5)
ax_poly.set_xticks(range(n_objects))
ax_poly.set_xticklabels(object_names, fontsize=12)
ax_poly.set_yticks(range(n_neurons))
ax_poly.set_yticklabels([f"N{i}" for i in range(n_neurons)], fontsize=10)

# Create heatmap for polysemanticity
polysemantic_img = ax_poly.imshow(
    np.zeros((n_neurons, n_objects)),
    cmap='viridis',
    aspect='auto',
    vmin=0,
    vmax=1
)

# Add a colorbar for the polysemanticity heatmap
cbar = fig.colorbar(polysemantic_img, ax=ax_poly)
cbar.set_label('Contribution Strength', fontsize=12)

# Highlight boxes for polysemantic neurons
highlight_patches = []
for neuron in polysemantic_neurons:
    rect = patches.Rectangle(
        (-0.5, neuron - 0.5),
        n_objects,
        1,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        alpha=0,  # Start invisible
        zorder=10
    )
    ax_poly.add_patch(rect)
    highlight_patches.append(rect)

# Animation update function
def update(frame):
    # Calculate which phase we're in
    cycle = frame // (n_frames // 4)
    phase_frame = frame % (n_frames // 4)
    progress = phase_frame / (n_frames // 4)
    
    # Different phases of the animation
    if cycle == 0:
        # Phase 1: Show each object's neuron activations sequentially
        current_obj = int(3 * progress)
        if current_obj >= n_objects:
            current_obj = n_objects - 1
            
        # Update bar plots
        for i, bars in enumerate(bar_containers):
            if i == current_obj:
                # Activate current object's neurons
                for j, bar in enumerate(bars):
                    bar.set_height(base_activations[i, j] * min(1, progress * 3 - i))
            else:
                # Reset other objects' neurons
                for bar in bars:
                    bar.set_height(0)
                    
        # Update polysemanticity heatmap
        poly_data = np.zeros((n_neurons, n_objects))
        if progress > 0.3:
            for i in range(current_obj + 1):
                poly_data[:, i] = base_activations[i]
        polysemantic_img.set_array(poly_data)
        
        # Hide highlight boxes
        for patch in highlight_patches:
            patch.set_alpha(0)
            
    elif cycle == 1:
        # Phase 2: Show all objects' neuron activations together
        # Update bar plots - show all activations
        for i, bars in enumerate(bar_containers):
            for j, bar in enumerate(bars):
                bar.set_height(base_activations[i, j] * progress)
                
        # Update polysemanticity heatmap - show all contributions
        poly_data = np.zeros((n_neurons, n_objects))
        for i in range(n_objects):
            poly_data[:, i] = base_activations[i] * progress
        polysemantic_img.set_array(poly_data)
        
        # Start highlighting polysemantic neurons
        for patch in highlight_patches:
            patch.set_alpha(progress * 0.5)
            
    elif cycle == 2:
        # Phase 3: Highlight polysemantic neurons
        # Keep bar plots fully visible
        for i, bars in enumerate(bar_containers):
            for j, bar in enumerate(bars):
                bar.set_height(base_activations[i, j])
                # Highlight polysemantic neurons in the bar plots
                if j in polysemantic_neurons:
                    # Pulse the polysemantic neurons
                    pulse = 0.7 + 0.3 * np.sin(progress * 2 * np.pi)
                    bar.set_alpha(pulse)
                    bar.set_edgecolor('red')
                    bar.set_linewidth(2)
                else:
                    bar.set_alpha(0.5)
                    bar.set_edgecolor('black')
                    bar.set_linewidth(1)
                
        # Keep polysemanticity heatmap fully visible
        poly_data = np.zeros((n_neurons, n_objects))
        for i in range(n_objects):
            poly_data[:, i] = base_activations[i]
        polysemantic_img.set_array(poly_data)
        
        # Pulse highlight boxes
        pulse = 0.5 + 0.5 * np.sin(progress * 2 * np.pi)
        for patch in highlight_patches:
            patch.set_alpha(pulse)
            
    else:
        # Phase 4: Show how polysemantic neurons enable efficient representation
        # Alternate between objects to show how the same neurons represent different things
        sub_cycle = int(progress * 6) % n_objects
        # Update bar plots - show one object at a time
        for i, bars in enumerate(bar_containers):
            for j, bar in enumerate(bars):
                if i == sub_cycle:
                    bar.set_height(base_activations[i, j])
                    # Highlight polysemantic neurons
                    if j in polysemantic_neurons:
                        bar.set_alpha(1.0)
                        bar.set_edgecolor('red')
                        bar.set_linewidth(2)
                    else:
                        bar.set_alpha(0.7)
                        bar.set_edgecolor('black')
                        bar.set_linewidth(1)
                else:
                    # Show ghost of other objects' activations for polysemantic neurons
                    if j in polysemantic_neurons:
                        bar.set_height(base_activations[i, j] * 0.3)
                        bar.set_alpha(0.3)
                    else:
                        bar.set_height(0)
                
        # Update polysemanticity heatmap - highlight current object
        poly_data = np.zeros((n_neurons, n_objects))
        for i in range(n_objects):
            if i == sub_cycle:
                poly_data[:, i] = base_activations[i]
            else:
                # Show ghost of other objects for polysemantic neurons
                for j in polysemantic_neurons:
                    poly_data[j, i] = base_activations[i, j] * 0.3
        polysemantic_img.set_array(poly_data)
        
        # Keep highlight boxes visible
        for patch in highlight_patches:
            patch.set_alpha(0.7)
    
    return bar_containers + [polysemantic_img] + highlight_patches

# Create animation
ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=50)

# Adjust layout
plt.tight_layout(rect=[0, 0.08, 1, 0.9])

# Save the animation
ani.save('polysemanticity_visualization.mp4', writer='ffmpeg', fps=15, dpi=150)

# Display the animation (will only work in interactive environments)
plt.show() 