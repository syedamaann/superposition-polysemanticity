


https://github.com/user-attachments/assets/4b53e7f7-7c94-480d-99b9-d6e1530cf319




https://github.com/user-attachments/assets/bfdb961b-7524-4aa1-af0b-970656fbc533


# Neural Network Superposition and Polysemanticity Visualization

This project visualizes the concepts of superposition and polysemanticity in neural networks through animations of neurons lighting up for different objects.

## Concepts Demonstrated

1. **Superposition**: The same set of neurons can represent multiple different objects or concepts. This is how neural networks efficiently use their limited resources.

2. **Polysemanticity**: Individual neurons contribute to multiple different representations, rather than each neuron being dedicated to a single concept.

## Visualizations

This project includes two different visualizations:

### 1. Enhanced Superposition Visualization (`enhanced_superposition_visualization.py`)

A detailed visualization that:
- Uses color coding to represent different objects (red, green, blue)
- Explicitly highlights shared neurons between different objects
- Shows a combined view that demonstrates superposition more clearly
- Provides a more intuitive understanding of how the same neurons participate in representing multiple objects
- Features clear explanations and improved spacing for better readability
- Uses a 7x7 grid layout for clearer visualization of neuron activations

### 2. Polysemanticity Visualization (`polysemanticity_visualization.py`)

A specialized visualization focusing on polysemanticity that:
- Uses bar charts to clearly show neuron activation levels for each object
- Includes a heatmap showing how each neuron contributes to different object representations
- Highlights polysemantic neurons (those that strongly contribute to multiple objects)
- Demonstrates through animation how the same neurons participate in representing different objects
- Progresses through multiple phases to build understanding of the concept
- Features improved spacing and text placement to avoid overlapping elements
- Includes detailed explanations of the concept being demonstrated

## How the Visualizations Work

The animations show neurons activating in different patterns for three different objects: a cat, a car, and a house. The key insights:

- Notice how some neurons light up for multiple objects (superposition)
- The activation patterns overlap but are distinct for each object
- There's a smooth transition between object representations
- Residual activations show how neural networks maintain information
- In the enhanced visualization, shared neurons are highlighted to make superposition more obvious
- The polysemanticity visualization specifically shows how individual neurons contribute to multiple representations

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- ffmpeg (for saving the animation)

## Installation

```bash
pip install -r requirements.txt
```

## Running the Visualizations

### Running Individual Visualizations

For the enhanced superposition visualization:
```bash
python enhanced_superposition_visualization.py
```

For the polysemanticity-focused visualization:
```bash
python polysemanticity_visualization.py
```

### Running All Visualizations

To run both visualizations in sequence:
```bash
python run_all_visualizations.py
```

This script will:
1. Run each visualization one after another
2. Generate both MP4 files
3. Display progress and completion information
4. Allow you to interrupt a visualization with Ctrl+C to move to the next one

Each script will generate an animation file (`.mp4`) and attempt to display the animation in an interactive window (if supported by your environment).

## Output Files

The following MP4 files will be generated:
- `enhanced_superposition.mp4` - Enhanced superposition visualization
- `polysemanticity_visualization.mp4` - Polysemanticity visualization

## Customization

You can modify the following parameters in the scripts:
- `n_neurons`: Number of neurons in the visualization
- `n_objects`: Number of different objects to represent
- `n_frames`: Number of frames in the animation
- `object_names`: Names of the objects being represented 
