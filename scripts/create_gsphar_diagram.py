#!/usr/bin/env python
"""
Script to create a diagram of the GSPHAR model architecture.
This script creates a simple diagram of the GSPHAR model architecture
using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_gsphar_diagram(output_dir='plots', filename='gsphar_model_diagram.png'):
    """
    Create a diagram of the GSPHAR model architecture.

    Args:
        output_dir (str): Directory to save the diagram.
        filename (str): Filename for the diagram.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set background color
    ax.set_facecolor('#f5f5f5')
    
    # Define colors
    input_color = '#d4f1f9'
    conv_color = '#ffcccc'
    spatial_color = '#ccffcc'
    linear_color = '#ffffcc'
    output_color = '#ffcc99'
    
    # Define box properties
    box_width = 2.0
    box_height = 0.6
    
    # Define positions
    x_start = 1
    y_start = 7
    
    # Draw input boxes
    ax.add_patch(patches.Rectangle((x_start, y_start), box_width, box_height, 
                                   facecolor=input_color, edgecolor='black', alpha=0.7))
    ax.text(x_start + box_width/2, y_start + box_height/2, 'Input x_lag1\n(batch, filter_size, input_dim)',
            ha='center', va='center', fontsize=10)
    
    ax.add_patch(patches.Rectangle((x_start, y_start - 1.5), box_width, box_height, 
                                   facecolor=input_color, edgecolor='black', alpha=0.7))
    ax.text(x_start + box_width/2, y_start - 1.5 + box_height/2, 'Input x_lag5\n(batch, filter_size, input_dim)',
            ha='center', va='center', fontsize=10)
    
    ax.add_patch(patches.Rectangle((x_start, y_start - 3), box_width, box_height, 
                                   facecolor=input_color, edgecolor='black', alpha=0.7))
    ax.text(x_start + box_width/2, y_start - 3 + box_height/2, 'Input x_lag22\n(batch, filter_size, input_dim)',
            ha='center', va='center', fontsize=10)
    
    # Draw FFT transformation
    ax.add_patch(patches.Rectangle((x_start + 2.5, y_start - 1.5), box_width, box_height, 
                                   facecolor='#e6e6fa', edgecolor='black', alpha=0.7))
    ax.text(x_start + 2.5 + box_width/2, y_start - 1.5 + box_height/2, 'FFT Transformation',
            ha='center', va='center', fontsize=10)
    
    # Draw arrows from inputs to FFT
    ax.arrow(x_start + box_width, y_start + box_height/2, 1.5, -1.5 + box_height/2, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(x_start + box_width, y_start - 1.5 + box_height/2, 1.5, 0, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(x_start + box_width, y_start - 3 + box_height/2, 1.5, 1.5 + box_height/2, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Draw Conv1D boxes
    ax.add_patch(patches.Rectangle((x_start + 5, y_start - 0.75), box_width, box_height, 
                                   facecolor=conv_color, edgecolor='black', alpha=0.7))
    ax.text(x_start + 5 + box_width/2, y_start - 0.75 + box_height/2, 'Conv1D (lag5)\n(filter_size, 1, 5)',
            ha='center', va='center', fontsize=10)
    
    ax.add_patch(patches.Rectangle((x_start + 5, y_start - 2.25), box_width, box_height, 
                                   facecolor=conv_color, edgecolor='black', alpha=0.7))
    ax.text(x_start + 5 + box_width/2, y_start - 2.25 + box_height/2, 'Conv1D (lag22)\n(filter_size, 1, 22)',
            ha='center', va='center', fontsize=10)
    
    # Draw arrows from FFT to Conv1D
    ax.arrow(x_start + 2.5 + box_width, y_start - 1.5 + box_height/2, 1.5, 0.75, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(x_start + 2.5 + box_width, y_start - 1.5 + box_height/2, 1.5, -0.75, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Draw Spatial Process box
    ax.add_patch(patches.Rectangle((x_start + 5, y_start - 3.75), box_width, box_height, 
                                   facecolor=spatial_color, edgecolor='black', alpha=0.7))
    ax.text(x_start + 5 + box_width/2, y_start - 3.75 + box_height/2, 'Spatial Process\nMLP (2→16→16→1)',
            ha='center', va='center', fontsize=10)
    
    # Draw arrow from FFT to Spatial Process
    ax.arrow(x_start + 2.5 + box_width, y_start - 1.5 + box_height/2, 1.5, -2.25, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Draw Linear Output boxes
    ax.add_patch(patches.Rectangle((x_start + 7.5, y_start - 1.5), box_width, box_height, 
                                   facecolor=linear_color, edgecolor='black', alpha=0.7))
    ax.text(x_start + 7.5 + box_width/2, y_start - 1.5 + box_height/2, 'Linear Output\nReal & Imaginary',
            ha='center', va='center', fontsize=10)
    
    # Draw arrows to Linear Output
    ax.arrow(x_start + 5 + box_width, y_start - 0.75 + box_height/2, 1.5, -0.75, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(x_start + 5 + box_width, y_start - 2.25 + box_height/2, 1.5, 0.75, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(x_start + 5 + box_width, y_start - 3.75 + box_height/2, 1.5, 2.25, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Draw Output box
    ax.add_patch(patches.Rectangle((x_start + 10, y_start - 1.5), box_width, box_height, 
                                   facecolor=output_color, edgecolor='black', alpha=0.7))
    ax.text(x_start + 10 + box_width/2, y_start - 1.5 + box_height/2, 'Output\n(batch, filter_size, output_dim)',
            ha='center', va='center', fontsize=10)
    
    # Draw arrow to Output
    ax.arrow(x_start + 7.5 + box_width, y_start - 1.5 + box_height/2, 1.5, 0, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add title
    ax.set_title('GSPHAR Model Architecture', fontsize=14, fontweight='bold')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set axis limits
    ax.set_xlim(0, 13)
    ax.set_ylim(2, 8)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=input_color, edgecolor='black', alpha=0.7, label='Input Layers'),
        patches.Patch(facecolor='#e6e6fa', edgecolor='black', alpha=0.7, label='FFT Transformation'),
        patches.Patch(facecolor=conv_color, edgecolor='black', alpha=0.7, label='Convolutional Layers'),
        patches.Patch(facecolor=spatial_color, edgecolor='black', alpha=0.7, label='Spatial Process MLP'),
        patches.Patch(facecolor=linear_color, edgecolor='black', alpha=0.7, label='Linear Output Layers'),
        patches.Patch(facecolor=output_color, edgecolor='black', alpha=0.7, label='Output Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.1),
              fancybox=True, shadow=True, ncol=3)
    
    # Add model details
    ax.text(0.5, 0.05, 
            'Total Parameters: 885\nFilter Size: 20\nInput Dimension: 3\nOutput Dimension: 1',
            ha='center', va='center', transform=fig.transFigure, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Model diagram saved to {os.path.join(output_dir, filename)}")
    
    # Close the figure
    plt.close()


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Create a diagram of the GSPHAR model architecture.')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save the diagram.')
    parser.add_argument('--filename', type=str, default='gsphar_model_diagram.png',
                        help='Filename for the diagram.')
    
    args = parser.parse_args()
    
    # Create model diagram
    create_gsphar_diagram(
        output_dir=args.output_dir,
        filename=args.filename
    )
    
    print("Model diagram creation completed successfully.")


if __name__ == '__main__':
    main()
