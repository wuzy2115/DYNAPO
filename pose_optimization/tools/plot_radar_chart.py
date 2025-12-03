import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_legend(labels, output_filename='legend.svg'):
    """
    Generates and saves a legend as a separate SVG file.

    Args:
        labels (list): A list of labels for the legend.
        output_filename (str): The name of the output SVG file.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    fig = plt.figure(figsize=(len(labels) * 1.5, 0.5)) # Adjust size based on number of labels
    ax = fig.add_subplot(111)

    # Create dummy lines with colors from the default cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    lines = []
    legend_labels = []
    for i, label in enumerate(labels):
        if label.lower() == 'skip':
            continue
        lines.append(plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2))
        legend_labels.append(label)

    # Create the legend horizontally
    ax.legend(lines, legend_labels, loc='center', ncol=len(legend_labels), frameon=False, fontsize=14)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(output_filename, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Legend saved to {output_filename}")


def plot_radar_chart(csv_files, labels, output_filename='radar_chart.svg', show_legend=True):
    """
    Generates and saves a radar chart from ATE, RTE, and RRE metrics
    found in the last column of given CSV files.

    Args:
        csv_files (list): A list of paths to the CSV files.
        labels (list): A list of labels for each CSV file for the plot legend.
        output_filename (str): The name of the output SVG file.
        show_legend (bool): If True, the legend is displayed on the plot.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    metrics = ['ATE', 'RTE', 'RRE']
    num_vars = len(metrics)

    # Read all data first to determine the scale for normalization
    all_values = []
    for file_path in csv_files:
        if file_path.lower() == 'skip':
            all_values.append([np.nan, np.nan, np.nan])
        else:
            df = pd.read_csv(file_path)
            values = df.iloc[0:3, -1].values.tolist()
            all_values.append(values)
    
    # Find the min and max of each metric across all files for rescaling
    min_metrics = np.nanmin(all_values, axis=0)
    max_metrics = np.nanmax(all_values, axis=0)
    
    # Determine the range, adding a buffer so points don't sit on the edge
    metric_range = max_metrics - min_metrics
    # Handle case where all values for a metric are the same
    metric_range[metric_range == 0] = 1.0
    
    scale_min = min_metrics - 0.1 * metric_range
    scale_max = max_metrics + 0.1 * metric_range
    plot_range = scale_max - scale_min
    # Ensure plot_range is not zero
    plot_range[plot_range == 0] = 1.0

    # Calculate angle for each axis, rotating to place ATE at the top-left
    angles = (np.linspace(0, 2 * np.pi, num_vars, endpoint=False) + 2 * np.pi / 3).tolist()
    
    # The plot is a circle, so we need to "complete the loop"
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Pre-calculate ATE ranks for custom label placement
    ate_values = [v[0] for v in all_values]
    ate_ranks = {original_index: rank for rank, original_index in enumerate(np.argsort(ate_values))}

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, (original_values, label) in enumerate(zip(all_values, labels)):
        if label.lower() == 'skip':
            continue

        # Rescale values for plotting
        rescaled_values = ((np.array(original_values) - scale_min) / plot_range).tolist()
        # Complete the loop
        rescaled_values += rescaled_values[:1]
        
        # Plot the data
        color = colors[i % len(colors)]
        ax.plot(angles, rescaled_values, linewidth=2, linestyle='solid', label=label, color=color)
        ax.fill(angles, rescaled_values, alpha=0.25, color=color)

        # --- Add Annotations with original values ---
        for j, (angle, norm_val, orig_val) in enumerate(zip(angles[:-1], rescaled_values[:-1], original_values)):
            metric_name = metrics[j]
            ha, va = 'center', 'center' # Default alignments

            if metric_name == 'ATE':
                # Alternate placement based on the sorted value of ATE
                rank = ate_ranks[i]
                ha = 'left' if rank % 2 == 0 else 'right'
            elif metric_name == 'RRE':
                # Alternate placement vertically (up/down)
                ha = 'center'
                va = 'bottom' if i % 2 == 0 else 'top'
            else: # RTE
                # Alternate placement horizontally based on file order
                ha = 'right' if i % 2 == 0 else 'left'

            ax.text(angle, norm_val + 0.05, f"{orig_val:.3f}", ha=ha, va=va, fontsize=10, fontweight='bold')


    # Style the plot
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # Remove default labels to place them manually

    # Manually place metric labels further from the center
    for angle, label in zip(angles[:-1], metrics):
        ax.text(angle, ax.get_ylim()[1] * 1.15, label,
                ha='center', va='center', fontsize=14, color='black')
                
    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    print(f"Radar chart successfully saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a radar chart or a standalone legend from experiment CSV files.')
    parser.add_argument('--csv_files', nargs='+', help='List of paths to CSV files. Required for radar chart.')
    parser.add_argument('--labels', nargs='+', required=True, help='List of labels for the legend.')
    parser.add_argument('--output_filename', type=str, default='radar_chart.svg', help='Output filename for the SVG.')
    parser.add_argument('--no_legend', action='store_true', help='Do not display the legend on the radar chart.')
    parser.add_argument('--legend_only', action='store_true', help='Only generate the legend as a separate file.')
    
    args = parser.parse_args()

    if args.legend_only:
        if not args.labels:
            raise ValueError("Labels are required to generate a legend.")
        plot_legend(args.labels, args.output_filename)
    else:
        if not args.csv_files:
            raise ValueError("CSV files are required to generate a radar chart.")
        if len(args.csv_files) != len(args.labels):
            raise ValueError("The number of CSV files must match the number of labels.")

        plot_radar_chart(args.csv_files, args.labels, args.output_filename, show_legend=args.no_legend)
