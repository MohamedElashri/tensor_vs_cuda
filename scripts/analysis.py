'''
python -m venv venv
source venv/bin/activate
pip install matplotlib pandas seaborn
python analysis.py -m 16    # For FP16 analysis only
python analysis.py -m 32    # For FP32 analysis only
python analysis.py -t       # For comparison analysis between FP16 and FP32
'''



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import glob
import re
import numpy as np
from matplotlib.patches import Rectangle

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compare Tensor Cores vs CUDA Cores for FP16 or FP32 models.")
parser.add_argument("-m", "--model", type=int, choices=[16, 32], required=False, help="Specify model precision (16 for FP16 or 32 for FP32)")
parser.add_argument("-t", "--together", action="store_true", help="Generate comparison plots between FP16 and FP32")
args = parser.parse_args()

if not args.together and args.model is None:
    parser.error("--model is required when not using --together")

# Define color schemes based on model precision
def get_implementation_colors(model_precision):
    if model_precision == 16:
        return {
            "Tensor Core": "#1b5e20",     # Dark green for Tensor Cores
            "CUDA Core FP16": "#1a237e"   # Dark blue for CUDA Cores
        }
    else:  # FP32
        return {
            "Tensor Core": "#1b5e20",    # Dark green for Tensor Cores
            "CUDA Core": "#1a237e"       # Dark blue for CUDA Cores
        }

# Colors for combined plot
def get_gpu_implementation_colors(model_precision):
    cuda_core_name = 'CUDA Core FP16' if model_precision == 16 else 'CUDA Core'
    return {
        'A100': {
            'Tensor Core': '#1b5e20',      # Dark green
            cuda_core_name: '#1a237e'  # Dark blue
        },
        'RTX3090': {
            'Tensor Core': '#4a148c',      # Dark purple
            cuda_core_name: '#311b92'  # Deep purple
        },
        'RTX2080TI': {
            'Tensor Core': '#b71c1c',      # Dark red
            cuda_core_name: '#880e4f'  # Dark pink
        }
    }


comparison_colors = {
    ('Tensor Core', 'FP16'): '#1b5e20',  # Dark green
    ('Tensor Core', 'FP32'): '#4CAF50',  # Light green
    ('CUDA Core', 'FP16'): '#1a237e',    # Dark blue
    ('CUDA Core', 'FP32'): '#3F51B5'     # Light blue
}

model_type_colors = {
    'Tensor Core': '#1b5e20',  # Dark green
    'CUDA Core': '#1a237e'     # Dark blue
}

comparison_markers = {
    ('Tensor Core', 'FP16'): 'o',
    ('Tensor Core', 'FP32'): 's',
    ('CUDA Core', 'FP16'): '^',
    ('CUDA Core', 'FP32'): 'D'
}


# Region colors for combined plot
region_colors = {
    'A100': '#2e7d32',      # Lighter green
    'RTX3090': '#673ab7',   # Lighter purple
    'RTX2080TI': '#d32f2f'  # Lighter red
}

# Set up LaTeX-style fonts and academic aesthetics for the plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (8, 6),
    "savefig.format": "pdf",
})

def parse_log_file(file_path, model_precision=None):
    match = re.search(r"(A100|RTX2080TI|RTX3090)", file_path, re.IGNORECASE)
    gpu_model = match.group(0) if match else "Unknown"

    with open(file_path, "r") as f:
        lines = f.readlines()

    model_type, avg_inference_time, throughput, repeat_factor = None, None, None, None

    for line in lines:
        line = line.strip()
        if "Repeat factor" in line:
            repeat_factor = int(line.split(":")[1].strip())
        elif "Model type" in line:
            model_type = line.split(":")[1].strip()
            # Standardize CUDA Core naming
            if "CUDA Core" in model_type:  # This will catch both "CUDA Core" and "CUDA Core FP16"
                model_type = "CUDA Core"
        elif "Average inference time" in line:
            avg_inference_time = float(line.split(":")[1].replace("ms", "").strip())
        elif "Throughput" in line:
            throughput = float(line.split(":")[1].replace("images/second", "").strip())

    return {
        "GPU": gpu_model,
        "Repeat Factor": repeat_factor,
        "Model Type": model_type,
        "Average Inference Time (ms)": avg_inference_time,
        "Throughput (images/second)": throughput
    }

def load_data_for_precision(precision):
    log_dirs = [f"logs_fp{precision}_sneezy", f"logs_fp{precision}_sleepy"]
    data = []
    for log_dir in log_dirs:
        for file_path in glob.glob(f"{log_dir}/*.log"):
            parsed_data = parse_log_file(file_path, precision)
            parsed_data['Precision'] = f'FP{precision}'  # Add precision information
            data.append(parsed_data)
    return pd.DataFrame(data)


def create_single_precision_plots(df, plots_dir, gpu_plots_dir, model_precision, implementation_colors, gpu_implementation_colors):
    gpu_models = ['A100', 'RTX2080TI', 'RTX3090']

    # Combined plot with regions
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create regions for each GPU
    gpu_regions = {}
    for gpu in gpu_models:
        gpu_data = df[df['GPU'] == gpu]
        if not gpu_data.empty:
            x = gpu_data['Number of Images'].values
            y = gpu_data['Throughput (images/second)'].values
            min_y = y.min() * 0.95
            max_y = y.max() * 1.05
            gpu_regions[gpu] = {
                'x_min': x.min(),
                'x_max': x.max(),
                'y_min': min_y,
                'y_max': max_y
            }

    # Draw regions
    alpha = 0.1
    for gpu, region in gpu_regions.items():
        rect = Rectangle((region['x_min'], region['y_min']),
                        region['x_max'] - region['x_min'],
                        region['y_max'] - region['y_min'],
                        facecolor=region_colors[gpu],
                        alpha=alpha,
                        label=f'{gpu} Region')
        ax.add_patch(rect)

    # Plot lines for each GPU and implementation
    for gpu in gpu_models:
        gpu_data = df[df['GPU'] == gpu]
        for model_type in gpu_data['Model Type'].unique():
            data = gpu_data[gpu_data['Model Type'] == model_type]
            color = gpu_implementation_colors[gpu][model_type]
            sns.lineplot(
                data=data,
                x="Number of Images",
                y="Throughput (images/second)",
                label=f"{gpu} - {model_type}",
                marker="o",
                color=color,
                linewidth=2,
                markersize=8,
                ax=ax
            )

    # Individual GPU plots
    for gpu_model in gpu_models:
        gpu_data = df[df['GPU'] == gpu_model]
        if gpu_data.empty:
            print(f"No data found for {gpu_model}. Skipping...")
            continue

        # Throughput Comparison
        plt.figure(figsize=(10, 7))
        sns.lineplot(data=gpu_data,
                    x="Number of Images",
                    y="Throughput (images/second)",
                    hue="Model Type",
                    marker="o",
                    palette=implementation_colors,
                    linewidth=2,
                    markersize=8)

        plt.title(rf"\textbf{{Tensor Cores vs CUDA Cores Throughput on {gpu_model} ({model_precision})}}")
        plt.xlabel(r"\textbf{Number of Images}")
        plt.ylabel(r"\textbf{Throughput (images/second)}")
        plt.legend(title=r"\textbf{Implementation}", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{gpu_plots_dir}/throughput_comparison_{gpu_model}_{model_precision.lower()}.pdf",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()

        # Inference Time Comparison
        plt.figure(figsize=(10, 7))
        sns.lineplot(data=gpu_data,
                    x="Number of Images",
                    y="Average Inference Time (ms)",
                    hue="Model Type",
                    marker="o",
                    palette=implementation_colors,
                    linewidth=2,
                    markersize=8)

        plt.title(rf"\textbf{{Tensor Cores vs CUDA Cores Inference Time on {gpu_model} ({model_precision})}}")
        plt.xlabel(r"\textbf{Number of Images}")
        plt.ylabel(r"\textbf{Average Inference Time (ms)}")
        plt.legend(title=r"\textbf{Implementation}", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{gpu_plots_dir}/inference_time_comparison_{gpu_model}_{model_precision.lower()}.pdf",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()

def create_comparison_plots(df, plots_dir, gpu_plots_dir):
    # Combined throughput comparison across precisions
    plt.figure(figsize=(14, 8))

    # Plot each combination separately for better control
    for model_type in df['Model Type'].unique():
        for precision in df['Precision'].unique():
            data = df[(df['Model Type'] == model_type) & (df['Precision'] == precision)]
            plt.plot(data['Number of Images'], data['Throughput (images/second)'],
                    label=f'{model_type} - {precision}',
                    color=comparison_colors[(model_type, precision)],
                    marker=comparison_markers[(model_type, precision)],
                    markersize=8,
                    linewidth=2)

    plt.title(r"\textbf{Performance Comparison: FP16 vs FP32 Implementations}")
    plt.xlabel(r"\textbf{Batch Size (Number of Images)}")
    plt.ylabel(r"\textbf{Processing Speed (Images/second)}")
    plt.legend(title=r"\textbf{Implementation - Precision}", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{plots_dir}/throughput_comparison_fp16_vs_fp32.pdf",
                bbox_inches="tight",
                dpi=300)
    plt.close()

    # Create plot_data for bar plots
    plot_data = df.copy()
    plot_data['Number of Images (×10,000)'] = plot_data['Number of Images'] / 10000

    # For bar plots, modify the title and labels to be more descriptive:
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=plot_data,
        x="Number of Images (×10,000)",
        y="Throughput (images/second)",
        hue="Model Type",
        palette=model_type_colors,  # Use the simple palette here
        alpha=0.8
    )

    plt.title(r"\textbf{Average Processing Speed by Implementation and Precision}")
    plt.xlabel(r"\textbf{Batch Size (×10,000 Images)}")
    plt.ylabel(r"\textbf{Average Processing Speed (Images/second)}")

    plt.legend(title=r"\textbf{Implementation - Precision}",
              bbox_to_anchor=(1.05, 1),
              loc='upper left')

    plt.savefig(f"{plots_dir}/average_processing_speed.pdf",
                bbox_inches="tight",
                dpi=300)
    plt.close()

    # Per-GPU comparison plots
    for gpu in df['GPU'].unique():
        gpu_data = df[df['GPU'] == gpu]
        plt.figure(figsize=(14, 8))

        # Line plot for this GPU
        for model_type in gpu_data['Model Type'].unique():
            for precision in gpu_data['Precision'].unique():
                data = gpu_data[(gpu_data['Model Type'] == model_type) &
                              (gpu_data['Precision'] == precision)]
                plt.plot(data['Number of Images'], data['Throughput (images/second)'],
                        label=f'{model_type} - {precision}',
                        color=comparison_colors[(model_type, precision)],
                        marker=comparison_markers[(model_type, precision)],
                        markersize=8,
                        linewidth=2)

        plt.title(rf"\textbf{{Performance Comparison on {gpu}: FP16 vs FP32}}")
        plt.xlabel(r"\textbf{Batch Size (Number of Images)}")
        plt.ylabel(r"\textbf{Processing Speed (Images/second)}")
        plt.legend(title=r"\textbf{Implementation - Precision}",
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{gpu_plots_dir}/throughput_comparison_{gpu}_fp16_vs_fp32.pdf",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()

        # Bar plot for this GPU
        gpu_plot_data = plot_data[plot_data['GPU'] == gpu]
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=gpu_plot_data,
            x="Number of Images (×10,000)",
            y="Throughput (images/second)",
            hue="Model Type",
            palette=model_type_colors,  # Use the simple palette here
            alpha=0.8
        )

        plt.title(rf"\textbf{{Average Processing Speed on {gpu} by Implementation and Precision}}")
        plt.xlabel(r"\textbf{Batch Size (×10,000 Images)}")
        plt.ylabel(r"\textbf{Average Processing Speed (Images/second)}")
        plt.legend(title=r"\textbf{Implementation - Precision}",
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add explanatory text
        plt.figtext(0.02, -0.1,
                   f"This plot shows the average processing speed on {gpu} for each implementation and precision level\n" +
                   "across different batch sizes. Higher bars indicate better performance (more images processed per second).",
                   wrap=True, horizontalalignment='left', fontsize=10)

        plt.savefig(f"{gpu_plots_dir}/average_processing_speed_{gpu}.pdf",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()

def print_statistics(df, together=False):
    if together:
        print("\nComparison Statistics between FP16 and FP32:")
        print("=" * 50)
        for gpu in df['GPU'].unique():
            print(f"\nGPU: {gpu}")
            print("-" * 30)
            gpu_data = df[df['GPU'] == gpu]
            for metric in ['Throughput (images/second)', 'Average Inference Time (ms)']:
                print(f"\n{metric}:")
                summary = gpu_data.groupby(['Precision', 'Model Type'])[metric].agg(['mean', 'std', 'min', 'max'])
                print(summary)
    else:
        model_precision = f"FP{args.model}"
        print(f"\nSummary Statistics for {model_precision} Model:")
        print("=" * 50)
        for gpu in df['GPU'].unique():
            print(f"\nGPU: {gpu}")
            print("-" * 30)
            gpu_data = df[df['GPU'] == gpu]
            for metric in ['Throughput (images/second)', 'Average Inference Time (ms)']:
                print(f"\n{metric}:")
                summary = gpu_data.groupby('Model Type')[metric].agg(['mean', 'std', 'min', 'max'])
                print(summary)

# Main execution
if args.together:
    # Load data for both precisions
    df_fp16 = load_data_for_precision(16)
    df_fp32 = load_data_for_precision(32)
    df = pd.concat([df_fp16, df_fp32], ignore_index=True)
    plots_dir = "plots_comparison"
else:
    # Original single-precision logic
    model_precision = f"FP{args.model}"
    df = load_data_for_precision(args.model)
    plots_dir = f"plots_{model_precision.lower()}"

os.makedirs(plots_dir, exist_ok=True)
gpu_plots_dir = os.path.join(plots_dir, "GPUs")
os.makedirs(gpu_plots_dir, exist_ok=True)

# Process data
df['Number of Images'] = df['Repeat Factor'] * 10000
grouped = df.groupby(["GPU", "Number of Images", "Model Type", "Precision"] if args.together else ["GPU", "Number of Images", "Model Type"]).mean().reset_index()

if args.together:
    create_comparison_plots(grouped, plots_dir, gpu_plots_dir)
else:
    implementation_colors = get_implementation_colors(args.model)
    gpu_implementation_colors = get_gpu_implementation_colors(args.model)
    create_single_precision_plots(grouped, plots_dir, gpu_plots_dir, f"FP{args.model}", implementation_colors, gpu_implementation_colors)

# Print statistics
print_statistics(grouped, args.together)

# Bar plots comparing average throughput - One per GPU and one combined
plt.figure(figsize=(12, 7))
ax = plt.gca()

# Create a copy of the data with scaled x-axis values
plot_data = grouped.copy()
plot_data['Number of Images (×10,000)'] = plot_data['Number of Images'] / 10000

if args.together:
    # Plot combined data with both precisions
    sns.barplot(
        data=plot_data,
        x="Number of Images (×10,000)",
        y="Throughput (images/second)",
        hue="Model Type",
        palette="deep",
        ax=ax
    )
    plt.title(r"\textbf{Average Throughput: FP16 vs FP32 Comparison}")
else:
    # Plot single precision data
    sns.barplot(
        data=plot_data,
        x="Number of Images (×10,000)",
        y="Throughput (images/second)",
        hue="Model Type",
        palette=implementation_colors,
        ax=ax
    )
    plt.title(rf"\textbf{{Average Throughput: Tensor Cores vs CUDA Cores ({f'FP{args.model}'})}}")

plt.xlabel(r"\textbf{Number of Images (×10,000)}")
plt.ylabel(r"\textbf{Throughput (images/second)}")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)

# Manually set legend
if args.together:
    plt.legend(title=r"\textbf{Implementation and Precision}", bbox_to_anchor=(1.05, 1), loc='upper left')
else:
    handles = [plt.Rectangle((0,0),1,1, color=color)
              for model, color in implementation_colors.items()]
    labels = list(implementation_colors.keys())
    ax.legend(handles, labels,
             title=r"\textbf{Implementation}",
             bbox_to_anchor=(1.05, 1),
             loc='upper left')

plt.savefig(f"{plots_dir}/average_throughput_comparison"
            f"{'_fp16_vs_fp32' if args.together else f'_fp{args.model}'}"
            ".pdf",
            bbox_inches="tight",
            dpi=300)
plt.close()

# Per-GPU bar plots
for gpu_model in df['GPU'].unique():
    gpu_data = plot_data[plot_data['GPU'] == gpu_model]
    if not gpu_data.empty:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        if args.together:
            sns.barplot(
                data=gpu_data,
                x="Number of Images (×10,000)",
                y="Throughput (images/second)",
                hue="Model Type",
                palette="deep",
                ax=ax
            )
            plt.title(rf"\textbf{{Average Throughput on {gpu_model}: FP16 vs FP32}}")
        else:
            sns.barplot(
                data=gpu_data,
                x="Number of Images (×10,000)",
                y="Throughput (images/second)",
                hue="Model Type",
                palette=implementation_colors,
                ax=ax
            )
            plt.title(rf"\textbf{{Average Throughput on {gpu_model}: Tensor Cores vs CUDA Cores ({f'FP{args.model}'})}}")

        plt.xlabel(r"\textbf{Number of Images (×10,000)}")
        plt.ylabel(r"\textbf{Throughput (images/second)}")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        if args.together:
            plt.legend(title=r"\textbf{Implementation and Precision}", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            handles = [plt.Rectangle((0,0),1,1, color=color)
                      for model, color in implementation_colors.items()]
            labels = list(implementation_colors.keys())
            ax.legend(handles, labels,
                     title=r"\textbf{Implementation}",
                     bbox_to_anchor=(1.05, 1),
                     loc='upper left')

            plt.savefig(f"{gpu_plots_dir}/average_throughput_{gpu_model}"
                        f"{'_fp16_vs_fp32' if args.together else f'_fp{args.model}'}"
                        ".pdf",
                        bbox_inches="tight",
                        dpi=300)
            plt.close()

# Reset Seaborn style
sns.reset_defaults()
