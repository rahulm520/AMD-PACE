# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os

import matplotlib  # Import matplotlib before importing pyplot

matplotlib.use("Agg")  # Use Agg backend to generate images without a display

import numpy as np
import matplotlib.colors as mcolors  # Import for more color options
import matplotlib.pyplot as plt
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_INFO, PACE_LLM_WARNING

from datastructs import BenchmarkResultsList


def create_comparison_bar_graph(
    data: BenchmarkResultsList, output_file_prefix: os.PathLike
):
    """
    Generates bar charts to compare frameworks

    Args:
        data (dict): Benchmark results data
        output_file_prefix (os.PathLike): Output directory to save the generated charts
    """

    PACE_LLM_ASSERT(
        isinstance(data, BenchmarkResultsList),
        f"data must be a BenchmarkResultsList object, got {type(data)}",
    )
    data = data.to_dict()  # Convert to dictionary
    model_name = data.get("model_name", "Unknown Model")
    benchmark_results = data.get("benchmark_results", [])
    PACE_LLM_ASSERT(benchmark_results, "No benchmark results found in the input data.")

    frameworks = [result["framework"] for result in benchmark_results]

    metric_configs = {
        "average_gen_time": {"ylabel": "Average Generation Time (seconds)"},
        "average_latency_per_token": {"ylabel": "Average Latency (seconds)"},
        "total_tps": {"ylabel": "Total TPS (tokens/second)"},
        "output_tps": {"ylabel": "Output TPS (tokens/second)"},
        "average_ttft": {"ylabel": "Average TTFT (seconds)"},
    }

    # Color palette for the bars
    colors = list(mcolors.TABLEAU_COLORS.keys())

    for metric_key, config in metric_configs.items():
        metric_values = [result["metrics"][metric_key] for result in benchmark_results]

        plt.figure(figsize=(12, 7))
        bars = plt.bar(frameworks, metric_values, color=colors[: len(frameworks)])

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.005,  # Adjust vertical position for label
                round(yval, 3),  # Display value rounded to 3 decimal places
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.xlabel("Frameworks", fontsize=12)
        plt.ylabel(config["ylabel"], fontsize=12)
        plt.title(
            f"{model_name} - Framework Comparison for {metric_key}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Generate filename and save the chart
        filepath = f"{output_file_prefix}_{metric_key}.png"
        plt.savefig(filepath)  # Save the figure to the file path
        PACE_LLM_INFO(f"Chart for {metric_key} saved to {filepath}")
        plt.close()


def create_comparison_line_graph(
    data: BenchmarkResultsList, output_file_prefix: os.PathLike
):

    PACE_LLM_ASSERT(
        isinstance(data, BenchmarkResultsList),
        f"data must be a BenchmarkResultsList object, got {type(data)}",
    )
    data = data.to_dict()  # Convert to dictionary
    model_name = data.get("model_name", "Unknown Model")
    benchmark_results = data.get("benchmark_results", [])
    PACE_LLM_ASSERT(benchmark_results, "No benchmark results found in the input data.")

    metrics_config = {
        "metrics": {
            "time_per_tokens": {
                "ylabel": "Per-Token Time (seconds)",
                "xlabel": "Token Index (Generated Token Number)",
            },
        },
        "system_metrics": {
            "cpu_usage": {
                "ylabel": "CPU Usage (%)",
                "xlabel": "Time (seconds)",
            },
            "ram_usage": {
                "ylabel": "RAM Usage (MB)",
                "xlabel": "Time (seconds)",
            },
        },
    }

    for metric_type, metric_list in metrics_config.items():
        for metric_key, config in metric_list.items():
            plt.figure(figsize=(12, 7))
            for result in benchmark_results:
                framework = result["framework"]
                if metric_type not in result or result[metric_type] is None:
                    PACE_LLM_WARNING(
                        f"No {metric_key} available for {framework}. Skipping..."
                    )
                    break
                values = result[metric_type].get(metric_key)
                if not values:
                    PACE_LLM_WARNING(
                        f"No {metric_key} found for {framework}. Skipping..."
                    )
                    break

                # Smooth the data using a moving average window
                window_size_heuristic = max(
                    int(np.sqrt(len(values))), 2
                )  # Ensure window size is at least 2

                values_np = np.array(values)
                cumulative_sum = np.cumsum(values_np)
                smoothed_data = np.empty_like(values_np, dtype=float)

                for i in range(len(values_np)):
                    if i < window_size_heuristic:
                        smoothed_data[i] = cumulative_sum[i] / (i + 1)
                    else:
                        smoothed_data[i] = (
                            cumulative_sum[i]
                            - cumulative_sum[i - window_size_heuristic]
                        ) / window_size_heuristic

                smoothed_data = smoothed_data.tolist()
                indices = list(range(len(smoothed_data)))

                plt.plot(
                    indices,
                    smoothed_data,
                    scalex=True,
                    scaley=True,
                    label=framework,
                    marker="o",
                    linestyle="-",
                    markersize=1,
                )
            else:
                plt.title(
                    f"{model_name} - {metric_key}",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.xlabel(config["xlabel"], fontsize=12)
                plt.ylabel(config["ylabel"], fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, which="both", linestyle="--", linewidth=0.5)
                plt.tight_layout()

                # Generate filename and save the chart
                filepath = f"{output_file_prefix}_{metric_key}.png"
                plt.savefig(filepath)
                PACE_LLM_INFO(f"Chart for {metric_key} saved to {filepath}")
                plt.close()
