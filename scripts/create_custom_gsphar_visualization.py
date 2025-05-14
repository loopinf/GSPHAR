#!/usr/bin/env python
"""
Script to create a custom visualization of the GSPHAR model architecture.
This script creates a detailed visualization of the GSPHAR model architecture
with tensor shapes at each step of the forward pass.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.models import GSPHAR


def create_custom_visualization(batch_size=32, filter_size=20, input_dim=22, output_dim=1,
                               output_dir='plots', filename='gsphar_custom_visualization.png'):
    """
    Create a custom visualization of the GSPHAR model architecture.

    Args:
        batch_size (int): Batch size for the input.
        filter_size (int): Filter size for the model.
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        output_dir (str): Directory to save the visualization.
        filename (str): Filename for the visualization.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Set background color
    ax.set_facecolor('#f8f8f8')
    
    # Define colors
    colors = {
        'input': '#d4f1f9',
        'fft': '#e6e6fa',
        'spectral': '#f0e68c',
        'conv': '#ffcccc',
        'spatial': '#ccffcc',
        'concat': '#ffffcc',
        'linear': '#ffcc99',
        'complex': '#ff9999',
        'output': '#ff8c00'
    }
    
    # Define box properties
    box_width = 3.0
    box_height = 0.8
    
    # Define positions
    x_start = 1
    y_start = 10
    
    # Helper function to draw a box with text
    def draw_box(x, y, width, height, color, text, fontsize=9):
        ax.add_patch(patches.Rectangle((x, y), width, height, 
                                       facecolor=color, edgecolor='black', alpha=0.7))
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize)
    
    # Helper function to draw an arrow
    def draw_arrow(x1, y1, x2, y2, color='black'):
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.15, head_length=0.15, 
                 fc=color, ec=color, length_includes_head=True)
    
    # Draw input boxes
    draw_box(x_start, y_start, box_width, box_height, colors['input'], 
             f'Input x_lag1\nShape: ({batch_size}, {filter_size}, {input_dim})')
    draw_box(x_start, y_start - 2, box_width, box_height, colors['input'], 
             f'Input x_lag5\nShape: ({batch_size}, {filter_size}, {input_dim})')
    draw_box(x_start, y_start - 4, box_width, box_height, colors['input'], 
             f'Input x_lag22\nShape: ({batch_size}, {filter_size}, {input_dim})')
    
    # Draw FFT transformation boxes
    draw_box(x_start + 4, y_start, box_width, box_height, colors['fft'], 
             f'FFT(x_lag1)\nShape: ({batch_size}, {filter_size}, {input_dim})')
    draw_box(x_start + 4, y_start - 2, box_width, box_height, colors['fft'], 
             f'FFT(x_lag5)\nShape: ({batch_size}, {filter_size}, {input_dim})')
    draw_box(x_start + 4, y_start - 4, box_width, box_height, colors['fft'], 
             f'FFT(x_lag22)\nShape: ({batch_size}, {filter_size}, {input_dim})')
    
    # Draw arrows from inputs to FFT
    draw_arrow(x_start + box_width, y_start + box_height/2, x_start + 4, y_start + box_height/2)
    draw_arrow(x_start + box_width, y_start - 2 + box_height/2, x_start + 4, y_start - 2 + box_height/2)
    draw_arrow(x_start + box_width, y_start - 4 + box_height/2, x_start + 4, y_start - 4 + box_height/2)
    
    # Draw spectral components boxes
    draw_box(x_start + 8, y_start + 0.5, box_width, box_height/2, colors['spectral'], 
             f'x_lag1_spectral_real\nShape: ({batch_size}, {filter_size}, {input_dim})')
    draw_box(x_start + 8, y_start, box_width, box_height/2, colors['spectral'], 
             f'x_lag1_spectral_imag\nShape: ({batch_size}, {filter_size}, {input_dim})')
    
    draw_box(x_start + 8, y_start - 1.5, box_width, box_height/2, colors['spectral'], 
             f'x_lag5_spectral_real\nShape: ({batch_size}, {filter_size}, {input_dim})')
    draw_box(x_start + 8, y_start - 2, box_width, box_height/2, colors['spectral'], 
             f'x_lag5_spectral_imag\nShape: ({batch_size}, {filter_size}, {input_dim})')
    
    draw_box(x_start + 8, y_start - 3.5, box_width, box_height/2, colors['spectral'], 
             f'x_lag22_spectral_real\nShape: ({batch_size}, {filter_size}, {input_dim})')
    draw_box(x_start + 8, y_start - 4, box_width, box_height/2, colors['spectral'], 
             f'x_lag22_spectral_imag\nShape: ({batch_size}, {filter_size}, {input_dim})')
    
    # Draw arrows from FFT to spectral components
    draw_arrow(x_start + 4 + box_width, y_start + box_height/2, x_start + 8, y_start + 0.5 + box_height/4)
    draw_arrow(x_start + 4 + box_width, y_start + box_height/2, x_start + 8, y_start + box_height/4)
    
    draw_arrow(x_start + 4 + box_width, y_start - 2 + box_height/2, x_start + 8, y_start - 1.5 + box_height/4)
    draw_arrow(x_start + 4 + box_width, y_start - 2 + box_height/2, x_start + 8, y_start - 2 + box_height/4)
    
    draw_arrow(x_start + 4 + box_width, y_start - 4 + box_height/2, x_start + 8, y_start - 3.5 + box_height/4)
    draw_arrow(x_start + 4 + box_width, y_start - 4 + box_height/2, x_start + 8, y_start - 4 + box_height/4)
    
    # Draw Conv1D boxes
    # Note: The output size will be (batch_size, filter_size, input_dim - kernel_size + 1)
    conv5_output_dim = input_dim - 5 + 1
    conv22_output_dim = input_dim - 22 + 1
    
    draw_box(x_start + 12, y_start - 1.5, box_width, box_height/2, colors['conv'], 
             f'Conv1D (lag5) Real\nShape: ({batch_size}, {filter_size}, {conv5_output_dim})')
    draw_box(x_start + 12, y_start - 2, box_width, box_height/2, colors['conv'], 
             f'Conv1D (lag5) Imag\nShape: ({batch_size}, {filter_size}, {conv5_output_dim})')
    
    draw_box(x_start + 12, y_start - 3.5, box_width, box_height/2, colors['conv'], 
             f'Conv1D (lag22) Real\nShape: ({batch_size}, {filter_size}, {conv22_output_dim})')
    draw_box(x_start + 12, y_start - 4, box_width, box_height/2, colors['conv'], 
             f'Conv1D (lag22) Imag\nShape: ({batch_size}, {filter_size}, {conv22_output_dim})')
    
    # Draw arrows to Conv1D
    draw_arrow(x_start + 8 + box_width, y_start - 1.5 + box_height/4, x_start + 12, y_start - 1.5 + box_height/4)
    draw_arrow(x_start + 8 + box_width, y_start - 2 + box_height/4, x_start + 12, y_start - 2 + box_height/4)
    draw_arrow(x_start + 8 + box_width, y_start - 3.5 + box_height/4, x_start + 12, y_start - 3.5 + box_height/4)
    draw_arrow(x_start + 8 + box_width, y_start - 4 + box_height/4, x_start + 12, y_start - 4 + box_height/4)
    
    # Draw Spatial Process boxes
    draw_box(x_start + 12, y_start, box_width, box_height, colors['spatial'], 
             f'Spatial Process\nInput: ({batch_size}, {filter_size}, 2)\nOutput: ({batch_size}, {filter_size}, 1)')
    
    # Draw arrows to Spatial Process
    draw_arrow(x_start + 8 + box_width, y_start + 0.5 + box_height/4, x_start + 12, y_start + box_height/2)
    draw_arrow(x_start + 8 + box_width, y_start + box_height/4, x_start + 12, y_start + box_height/2)
    
    # Draw Feature Concatenation boxes
    draw_box(x_start + 16, y_start - 1.5, box_width, box_height/2, colors['concat'], 
             f'Feature Concat (Real)\nShape: ({batch_size}, {filter_size}, 3)')
    draw_box(x_start + 16, y_start - 2, box_width, box_height/2, colors['concat'], 
             f'Feature Concat (Imag)\nShape: ({batch_size}, {filter_size}, 3)')
    
    # Draw arrows to Feature Concatenation
    draw_arrow(x_start + 12 + box_width, y_start + box_height/2, x_start + 16, y_start - 1.5 + box_height/4)
    draw_arrow(x_start + 12 + box_width, y_start - 1.5 + box_height/4, x_start + 16, y_start - 1.5 + box_height/4)
    draw_arrow(x_start + 12 + box_width, y_start - 3.5 + box_height/4, x_start + 16, y_start - 1.5 + box_height/4)
    
    draw_arrow(x_start + 12 + box_width, y_start + box_height/2, x_start + 16, y_start - 2 + box_height/4)
    draw_arrow(x_start + 12 + box_width, y_start - 2 + box_height/4, x_start + 16, y_start - 2 + box_height/4)
    draw_arrow(x_start + 12 + box_width, y_start - 4 + box_height/4, x_start + 16, y_start - 2 + box_height/4)
    
    # Draw Linear Output boxes
    draw_box(x_start + 20, y_start - 1.5, box_width, box_height/2, colors['linear'], 
             f'Linear Output (Real)\nShape: ({batch_size}, {filter_size}, {output_dim})')
    draw_box(x_start + 20, y_start - 2, box_width, box_height/2, colors['linear'], 
             f'Linear Output (Imag)\nShape: ({batch_size}, {filter_size}, {output_dim})')
    
    # Draw arrows to Linear Output
    draw_arrow(x_start + 16 + box_width, y_start - 1.5 + box_height/4, x_start + 20, y_start - 1.5 + box_height/4)
    draw_arrow(x_start + 16 + box_width, y_start - 2 + box_height/4, x_start + 20, y_start - 2 + box_height/4)
    
    # Draw Complex Reconstruction box
    draw_box(x_start + 24, y_start - 1.75, box_width, box_height, colors['complex'], 
             f'Complex Reconstruction\nShape: ({batch_size}, {filter_size}, {output_dim})')
    
    # Draw arrows to Complex Reconstruction
    draw_arrow(x_start + 20 + box_width, y_start - 1.5 + box_height/4, x_start + 24, y_start - 1.75 + box_height/2)
    draw_arrow(x_start + 20 + box_width, y_start - 2 + box_height/4, x_start + 24, y_start - 1.75 + box_height/2)
    
    # Draw Output box
    draw_box(x_start + 28, y_start - 1.75, box_width, box_height, colors['output'], 
             f'Output\nShape: ({batch_size}, {filter_size}, {output_dim})')
    
    # Draw arrow to Output
    draw_arrow(x_start + 24 + box_width, y_start - 1.75 + box_height/2, x_start + 28, y_start - 1.75 + box_height/2)
    
    # Add title
    ax.set_title('GSPHAR Model Architecture with Tensor Shapes', fontsize=16, fontweight='bold')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set axis limits
    ax.set_xlim(0, 32)
    ax.set_ylim(5, 12)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=color, edgecolor='black', alpha=0.7, label=name.capitalize())
        for name, color in colors.items()
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.1),
              fancybox=True, shadow=True, ncol=5)
    
    # Add model information
    ax.text(0.5, 0.05, 
            f'GSPHAR Model\nTotal Parameters: 885\nFilter Size: {filter_size}\nInput Dimension: {input_dim}\nOutput Dimension: {output_dim}',
            ha='center', va='center', transform=fig.transFigure, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Custom visualization saved to {os.path.join(output_dir, filename)}")
    
    # Close the figure
    plt.close()


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Create a custom visualization of the GSPHAR model architecture.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for the input.')
    parser.add_argument('--filter-size', type=int, default=20,
                        help='Filter size for the model.')
    parser.add_argument('--input-dim', type=int, default=22,
                        help='Input dimension.')
    parser.add_argument('--output-dim', type=int, default=1,
                        help='Output dimension.')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save the visualization.')
    parser.add_argument('--filename', type=str, default='gsphar_custom_visualization.png',
                        help='Filename for the visualization.')
    
    args = parser.parse_args()
    
    # Create custom visualization
    create_custom_visualization(
        batch_size=args.batch_size,
        filter_size=args.filter_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        output_dir=args.output_dir,
        filename=args.filename
    )
    
    print("Custom visualization completed successfully.")


if __name__ == '__main__':
    main()
