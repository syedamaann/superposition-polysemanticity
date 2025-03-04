import subprocess
import os
import time

def run_visualization(script_name, output_name):
    """Run a visualization script and wait for it to complete."""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}")
    
    # Run the script
    process = subprocess.Popen(['python', script_name])
    process.wait()
    
    # Check if the output file was created
    if os.path.exists(output_name):
        print(f"\n✅ Successfully created {output_name}")
        file_size = os.path.getsize(output_name) / (1024 * 1024)  # Convert to MB
        print(f"   File size: {file_size:.2f} MB")
    else:
        print(f"\n❌ Failed to create {output_name}")

def main():
    """Run all visualizations in sequence."""
    print("\n" + "="*80)
    print(" 🧠  Neural Network Superposition and Polysemanticity Visualizations  🧠 ")
    print("="*80 + "\n")
    
    # List of visualizations to run
    visualizations = [
        (
            "enhanced_superposition_visualization.py", 
            "enhanced_superposition.mp4"
        ),
        (
            "polysemanticity_visualization.py", 
            "polysemanticity_visualization.mp4"
        )
    ]
    
    # Run each visualization
    for script, output in visualizations:
        try:
            run_visualization(script, output)
            time.sleep(1)  # Brief pause between visualizations
        except KeyboardInterrupt:
            print("\nVisualization interrupted by user. Moving to next visualization...")
            continue
    
    print("\n" + "="*80)
    print(" 🎉  All visualizations completed!  🎉 ")
    print("="*80)
    
    print("\nOutput files:")
    for _, output in visualizations:
        if os.path.exists(output):
            file_size = os.path.getsize(output) / (1024 * 1024)  # Convert to MB
            print(f"- {output} ({file_size:.2f} MB)")

if __name__ == "__main__":
    main() 