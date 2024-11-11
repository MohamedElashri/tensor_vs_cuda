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
parser.add_argument("-m", "--model", type=int, choices=[16, 32], required=True, help="Specify model precision (16 for FP16 or 32 for FP32)")
args = parser.parse_args()

# Define paths based on model precision
model_precision = f"FP{args.model}"
log_dirs = [f"logs_fp{args.model}_sneezy", f"logs_fp{args.model}_sleepy"]

# Define color schemes based on model precision
if args.model == 16:
    implementation_colors = {
        "Tensor Core": "#1b5e20",     # Dark green for Tensor Cores
        "CUDA Core FP16": "#1a237e"   # Dark blue for CUDA Cores
    }
else:  # FP32
    implementation_colors = {
        "Tensor Core": "#1b5e20",    # Dark green for Tensor Cores
        "CUDA Core": "#1a237e"       # Dark blue for CUDA Cores
    }

# Colors for combined plot
gpu_implementation_colors = {
    'A100': {
        'Tensor Core': '#1b5e20',      # Dark green
        'CUDA Core FP16' if args.model == 16 else 'CUDA Core': '#1a237e'  # Dark blue
    },
    'RTX3090': {
        'Tensor Core': '#4a148c',      # Dark purple
        'CUDA Core FP16' if args.model == 16 else 'CUDA Core': '#311b92'  # Deep purple
    },
    'RTX2080TI': {
        'Tensor Core': '#b71c1c',      # Dark red
        'CUDA Core FP16' if args.model == 16 else 'CUDA Core': '#880e4f'  # Dark pink
    }
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

def parse_log_file(file_path):
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
            # Handle different naming conventions for CUDA Core
            if model_type == "CUDA Core" and args.model == 16:
                model_type = "CUDA Core FP16"
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

# Load and process data
data = []
for log_dir in log_dirs:
    for file_path in glob.glob(f"{log_dir}/*.log"):
        parsed_data = parse_log_file(file_path)
        data.append(parsed_data)

# Convert to DataFrame
df = pd.DataFrame(data)
df['Number of Images'] = df['Repeat Factor'] * 10000
grouped = df.groupby(["GPU", "Number of Images", "Model Type"]).mean().reset_index()

# Create output directories
plots_dir = f"plots_{model_precision.lower()}"
os.makedirs(plots_dir, exist_ok=True)
gpu_plots_dir = os.path.join(plots_dir, "GPUs")
os.makedirs(gpu_plots_dir, exist_ok=True)

# List of GPUs to analyze
gpu_models = ['A100', 'RTX2080TI', 'RTX3090']

# Combined plot with regions
fig, ax = plt.subplots(figsize=(12, 8))

# Create regions for each GPU
gpu_regions = {}
for gpu in gpu_models:
    gpu_data = grouped[grouped['GPU'] == gpu]
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
    gpu_data = grouped[grouped['GPU'] == gpu]
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

# Enhance the plot
ax.set_title(rf"\textbf{{Throughput Comparison Across GPUs: Tensor Cores vs CUDA Cores ({model_precision})}}", pad=20)
ax.set_xlabel(r"\textbf{Number of Images}", labelpad=10)
ax.set_ylabel(r"\textbf{Throughput (images/second)}", labelpad=10)

# Create custom legend
handles, labels = ax.get_legend_handles_labels()
region_handles = [Rectangle((0,0),1,1, facecolor=region_colors[gpu], alpha=alpha)
                 for gpu in gpu_models if gpu in gpu_regions]
region_labels = [f'{gpu} Region' for gpu in gpu_models if gpu in gpu_regions]

all_handles = handles + region_handles
all_labels = labels + region_labels

ax.legend(all_handles, all_labels,
         title=r"\textbf{GPU and Implementation}",
         bbox_to_anchor=(1.05, 1),
         loc='upper left',
         frameon=True,
         fancybox=True,
         shadow=True)

ax.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{plots_dir}/combined_throughput_{model_precision.lower()}.pdf",
            bbox_inches="tight",
            dpi=300)
plt.close()

# Individual GPU plots
for gpu_model in gpu_models:
    gpu_data = grouped[grouped['GPU'].str.contains(gpu_model, case=False)]
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

# Bar plots comparing average throughput - One per GPU and one combined
plt.figure(figsize=(12, 7))
ax = plt.gca()

# Create a copy of the data with scaled x-axis values
plot_data = grouped.copy()
plot_data['Number of Images (×10,000)'] = plot_data['Number of Images'] / 10000

# Plot combined data
sns.barplot(data=plot_data,
            x="Number of Images (×10,000)",
            y="Throughput (images/second)",
            hue="Model Type",
            palette=implementation_colors,
            ax=ax)

plt.title(rf"\textbf{{Average Throughput: Tensor Cores vs CUDA Cores ({model_precision})}}")
plt.xlabel(r"\textbf{Number of Images (×10,000)}")
plt.ylabel(r"\textbf{Throughput (images/second)}")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)

# Manually set legend
handles = [plt.Rectangle((0,0),1,1, color=color)
          for model, color in implementation_colors.items()]
labels = list(implementation_colors.keys())
ax.legend(handles, labels,
         title=r"\textbf{Implementation}",
         bbox_to_anchor=(1.05, 1),
         loc='upper left')

plt.savefig(f"{plots_dir}/average_throughput_comparison_{model_precision.lower()}.pdf",
            bbox_inches="tight",
            dpi=300)
plt.close()

# Per-GPU bar plots
for gpu_model in gpu_models:
    gpu_data = plot_data[plot_data['GPU'] == gpu_model]
    if not gpu_data.empty:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        sns.barplot(data=gpu_data,
                   x="Number of Images (×10,000)",
                   y="Throughput (images/second)",
                   hue="Model Type",
                   palette=implementation_colors,
                   ax=ax)

        plt.title(rf"\textbf{{Average Throughput on {gpu_model}: Tensor Cores vs CUDA Cores ({model_precision})}}")
        plt.xlabel(r"\textbf{Number of Images (×10,000)}")
        plt.ylabel(r"\textbf{Throughput (images/second)}")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Manually set legend
        handles = [plt.Rectangle((0,0),1,1, color=color)
                  for model, color in implementation_colors.items()]
        labels = list(implementation_colors.keys())
        ax.legend(handles, labels,
                 title=r"\textbf{Implementation}",
                 bbox_to_anchor=(1.05, 1),
                 loc='upper left')

        plt.savefig(f"{gpu_plots_dir}/average_throughput_comparison_{gpu_model}_{model_precision.lower()}.pdf",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()

# Print summary statistics
print(f"\nSummary Statistics for {model_precision} Model:")
print("=" * 50)
for gpu in gpu_models:
    gpu_data = grouped[grouped['GPU'] == gpu]
    if not gpu_data.empty:
        print(f"\nGPU: {gpu}")
        print("-" * 30)
        for metric in ['Throughput (images/second)', 'Average Inference Time (ms)']:
            print(f"\n{metric}:")
            summary = gpu_data.groupby('Model Type')[metric].agg(['mean', 'std', 'min', 'max'])
            print(summary)

# Reset Seaborn style
sns.reset_defaults()
