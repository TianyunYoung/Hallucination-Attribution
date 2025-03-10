import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import os

plt.rcParams['font.size'] = 14        # Default font size
plt.rcParams['axes.labelsize'] = 14       # Default axes label size
plt.rcParams['xtick.labelsize'] = 11       # Default x tick label size
plt.rcParams['ytick.labelsize'] = 14       # Default y tick label size
plt.rcParams['legend.fontsize'] = 14       # Default legend font size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--step-list", nargs='+', type=int)

    args = parser.parse_args()

    hal_heads_js_div_avg_list = []
    non_hal_heads_js_div_avg_list = []
    for step in args.step_list:
        hal_heads_js_div_all_samples = torch.load(os.path.join(args.output_path, f'step{step}', f'hal_heads_js_div.pth'), weights_only=True)
        non_hal_heads_js_div_all_samples = torch.load(os.path.join(args.output_path, f'step{step}', f'non_hal_heads_js_div.pth'), weights_only=True)

        hal_heads_js_div_avg = torch.mean(torch.tensor(hal_heads_js_div_all_samples))
        non_hal_heads_js_div_avg = torch.mean(torch.tensor(non_hal_heads_js_div_all_samples))
        hal_heads_js_div_avg_list.append(hal_heads_js_div_avg)
        non_hal_heads_js_div_avg_list.append(non_hal_heads_js_div_avg)
        print(f'step {step}: hal_heads_js_div: {hal_heads_js_div_avg}, non_hal_heads_js_div: {non_hal_heads_js_div_avg}')   

    x = np.arange(len(args.step_list))
    plt.figure()  # Adjusted size to better fit your example

    plt.plot(x, hal_heads_js_div_avg_list, '^-', color='blue', label='Hallucination Heads (Top 3)', markersize=10)  # Matched color code
    plt.plot(x, non_hal_heads_js_div_avg_list, '^-', color='red', label='Non-Hallucination Heads (Top 3)', markersize=10)  # Matched color code

    # Adding labels, title, and annotations
    plt.xlabel('Instruction Tuning Steps')
    plt.ylabel('JS Divergence with the Pre-trained Model')

    plt.ylim(0, 0.16)  # Adjust y-axis limits to match your plot
    plt.xticks(x, labels=[str(i) for i in args.step_list])

    # Add legend
    plt.legend(loc='best')
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)
    # Show the plot
    plt.tight_layout()
    print(f'{args.output_path}/js_div_in_training.png')
    plt.savefig(f'{args.output_path}/js_div_in_training.png')
    plt.savefig(f'{args.output_path}/js_div_in_training.pdf')






