import argparse
import asyncio
import json
import os
import random
import statistics as stats
import string
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from itertools import product
from typing import List, Optional, Type

import httpx
import matplotlib.pyplot as plt
import numpy as np
import replicate
from transformers import AutoTokenizer

import os
import importlib.util
from pathlib import Path

from load_tester.replicate_base_target import ReplicateBaseTargetType


REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


def get_target_type_from_name(target_canonical_name) -> Type[ReplicateBaseTargetType]:
    current_directory = Path(__file__).parent
    target_directory = current_directory / "target_types"
    for file_path in Path(target_directory).glob("*.py"):
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for _name, obj in module.__dict__.items():
            if isinstance(obj, type) and hasattr(obj, "canonical_name"):
                if getattr(obj, "canonical_name") == target_canonical_name:
                    return obj
    raise Exception(f"No Target Type matches name {target_canonical_name}")


def get_request_handler(target):
    if target not in ["triton", "cog-triton"]:
        return get_target_type_from_name(tar)


class Experiment:
    def __init__(self, experiment_dir, args, n_input_tokens, n_output_tokens, rate):
        self.experiment_dir = experiment_dir
        self.args = args
        self.n_input_tokens = n_input_tokens
        self.n_output_tokens = n_output_tokens
        self.rate = rate
        self.times = []
        self.failures = 0
        self.sstps = []
        self.start_end_times = []
        self.start_times = []
        self.returned_requests = []
        self.n_requests_made = 0
        self.n_requests_started = 0
        self.n_requests_completed = 0
        self.n_cog_already_running_prediction = 0
        self.time_to_first_token = []
        self.server_side_actual_tps = []
        self.server_side_actual_execution_time = []
        self.server_side_actual_time_to_first_token = []
        self.tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )
        self.duration = args.duration
        self.target = args.target
        self.target_type = args.target_type

        self.request_handler: ReplicateBaseTargetType = get_target_type_from_name(
            args.target_type
        )(experiment=self)

        if self.target not in ["cog-triton", "triton"]:

            self.headers = {
                "Authorization": f"Token {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json",
            }
            self.url = "https://api.replicate.com/v1/predictions"

        else:
            self.headers = {"Content-Type": "application/json"}
            self.url = (
                "http://localhost:8000/v2/models/ensemble/generate_stream"
                if self.target == "triton"
                else "http://localhost:5000/predictions"
            )

    def append_server_side_actual_tps(self, val):
        self.server_side_actual_tps.append(val)

    def append_server_side_actual_execution_time(self, val):
        self.server_side_actual_execution_time.append(val)

    def append_serverside_time_to_first_token(self, val):
        self.server_side_actual_time_to_first_token.append(val)

    def append_start_time(self, start_time):
        self.start_times.append(start_time)

    def append_latency(self, delta):
        self.times.append(delta)

    def append_start_end_times(self, start_time, end_time):
        self.start_end_times.append((start_time, end_time))

    def append_ttft(self, ttft):
        self.time_to_first_token.append(ttft)

    def append_sstp(self, sstp):
        self.sstps.append(sstp)

    def append_returned_request(self, request):
        self.returned_requests.append(request)

    def increment_requests_made(self):
        self.n_requests_made += 1

    def increment_requests_started(self):
        self.n_requests_started += 1

    async def perform_requests(self, client, url, headers):
        start_time = datetime.now()
        tasks = []
        mode = "batch" if self.rate > 1 else "rps"  # Example logic, adjust as necessary
        while (datetime.now() - start_time).total_seconds() < self.duration:
            if mode == "batch":
                tasks = [
                    asyncio.create_task(
                        self.request_handler.make_request(
                            prompt=self.get_prompt(),
                            max_number_tokens=self.n_output_tokens,
                        )
                    )
                    for _ in range(self.rate)
                ]
                await asyncio.gather(*tasks)
                await asyncio.sleep(0.5)  # Rate limit batches per second
            elif mode == "rps":
                await asyncio.sleep(1 / self.rate)  # Sleep to maintain the rate
                tasks.append(
                    asyncio.create_task(
                        self.request_handler.make_request(
                            prompt=self.get_prompt(),
                            max_number_tokens=self.n_output_tokens,
                        )
                    )
                )

        if tasks:
            await asyncio.gather(*tasks)

    def get_prompt(self):
        alphabet = string.ascii_lowercase  # Lowercase letters of the alphabet
        prompt = " ".join(random.choice(alphabet) for _ in range(self.n_input_tokens))
        return prompt

    async def run(self):
        print("*****" * 10)
        print(
            f"Starting experiment: rate={self.rate}, input_tokens={self.n_input_tokens}, output_tokens={self.n_output_tokens}"
        )
        print("*****" * 10)
        self.start_time = datetime.now()
        async with httpx.AsyncClient(timeout=300) as client:
            self.start_time = datetime.now()
            await self.perform_requests(client, self.url, self.headers)

    def generate_plots(self):
        target_for_title = (
            self.args.target
            if self.args.target in ["cog-triton", "triton"]
            else self.args.target[0:7]
        )
        plot_file_path_tps = os.path.join(
            self.experiment_dir, "tps_per_response_with_lines.png"
        )
        self.plot_metrics_with_lines(
            range(len(self.sstps)),
            self.sstps,
            "Response Number",
            "Single-Stream TPS",
            f"{target_for_title} Single-stream TPS per Response -- {self.args.unit}={self.args.rate}",
            plot_file_path_tps,
        )

        plot_file_path_latency = os.path.join(
            self.experiment_dir, "latency_per_response_with_lines.png"
        )

        self.plot_metrics_with_lines(
            range(len(self.times)),
            self.times,
            "Response Number",
            "Latency (seconds)",
            f"{target_for_title} Latency per Response -- {self.args.unit}={self.args.rate}",
            plot_file_path_latency,
        )

        if (
            self.args.target not in ["cog-triton", "triton"]
            and self.time_to_first_token
        ):
            plot_file_path_first_token = os.path.join(
                self.experiment_dir, "time_to_first_token_with_lines.png"
            )
            self.plot_metrics_with_lines(
                range(len(self.time_to_first_token)),
                self.time_to_first_token,
                "Request Number",
                "Time-to-First-Token (seconds)",
                f"{target_for_title} Time-to-First-Token per Request -- {self.args.unit}={self.args.rate}",
                plot_file_path_first_token,
            )

        # Define labels and titles for the plots

        # Define the main title and subtitles for the plots
        main_title = f"{target_for_title} -- {self.args.unit.capitalize()}={self.args.rate}, Duration={self.args.duration}s"
        subtitles = [
            "Single-stream TPS per Response",
            "Latency per Response",
            "Time-to-First-Token per Request",
        ]
        labels = [
            ("Response Number", "Single-Stream TPS"),
            ("Response Number", "Latency (seconds)"),
            ("Request Number", "Time-to-First-Token (seconds)"),
        ]

        # Ensure that the metrics for the third plot (time-to-first-token) are only added if applicable
        x3 = (
            range(len(self.time_to_first_token))
            if self.args.target not in ["cog-triton", "triton"]
            else []
        )
        y3 = (
            self.time_to_first_token
            if self.args.target not in ["cog-triton", "triton"]
            else []
        )

        # Call the plotting function
        self.plot_metrics_with_subplots(
            range(len(self.sstps)),
            self.sstps,
            range(len(self.times)),
            self.times,
            x3,
            y3,
            labels,
            subtitles,
            main_title,
            self.experiment_dir,
        )

    def plot_metrics_with_lines(self, x, y, x_label, y_label, title, file_name):
        plt.figure(figsize=(10, 6))

        # Sort the x and y values based on x to connect them correctly
        sorted_indices = sorted(range(len(x)), key=lambda i: x[i])
        sorted_x = [x[i] for i in sorted_indices]
        sorted_y = [y[i] for i in sorted_indices]

        # Plot points and connect them with a line
        plt.scatter(sorted_x, sorted_y, alpha=0.5)  # Plot points
        plt.plot(
            sorted_x, sorted_y, "-o", label="Data", color="blue"
        )  # Connect points with line

        # Add a dotted line for the median
        median_value = stats.median(y)
        plt.axhline(
            y=median_value,
            color="r",
            linestyle="--",
            label=f"Median: {median_value:.3f}",
        )

        # Add titles and labels
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()  # Show legend

        # Save and close
        plt.savefig(file_name)
        plt.close()

    def plot_metrics_with_subplots(
        self, x1, y1, x2, y2, x3, y3, labels, subtitles, main_title, experiment_dir
    ):
        """
        Plot metrics with subplots, showing median, mean, p99, and p01.

        Parameters:
            x1, y1: Data for the first subplot.
            x2, y2: Data for the second subplot.
            x3, y3: Data for the third subplot.
            labels: x and y labels for the plots.
            subtitles: Subtitles for the subplots.
            main_title: Main title for the entire figure.
            experiment_dir: Directory to save the plot.
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(main_title, fontsize=14)

        for i, (x, y, subtitle) in enumerate(
            zip([x1, x2, x3], [y1, y2, y3], subtitles)
        ):
            sorted_indices = sorted(range(len(x)), key=lambda k: x[k])
            sorted_x = [x[k] for k in sorted_indices]
            sorted_y = [y[k] for k in sorted_indices]

            axs[i].scatter(sorted_x, sorted_y, alpha=0.5)
            axs[i].plot(sorted_x, sorted_y, "-o", color="blue")

            median_value = stats.median(sorted_y)
            mean_value = stats.mean(sorted_y)
            p99_value = np.percentile(sorted_y, 99)
            p01_value = np.percentile(sorted_y, 1)

            axs[i].axhline(
                y=median_value,
                color="r",
                linestyle="--",
                label=f"Median: {median_value:.3f}",
            )
            axs[i].axhline(
                y=mean_value, color="g", linestyle="--", label=f"Mean: {mean_value:.3f}"
            )
            axs[i].axhline(
                y=p99_value, color="m", linestyle="--", label=f"P99: {p99_value:.3f}"
            )
            axs[i].axhline(
                y=p01_value, color="y", linestyle="--", label=f"P01: {p01_value:.3f}"
            )

            axs[i].set_title(subtitle)
            axs[i].set_xlabel(labels[i][0])
            axs[i].set_ylabel(labels[i][1])
            axs[i].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(experiment_dir, "combined_metrics_with_stats.png"))
        plt.close()

    def calculate_concurrency(self, start_end_times, duration, start_time):
        """
        Calculate concurrency levels for each 5ms interval of the test duration.
        """
        # Calculate the total number of intervals in the given duration
        # Each interval is 5ms, so multiply duration by 200 to get the number of intervals in one second
        total_intervals = duration * 200
        concurrency_levels = []

        # Iterate over each 5ms interval
        for interval in range(total_intervals):
            interval_start = start_time + timedelta(milliseconds=interval * 5)
            interval_end = interval_start + timedelta(milliseconds=5)

            # Count requests that were active during this interval
            concurrency = sum(
                1
                for start, end in start_end_times
                if start < interval_end and end > interval_start
            )

            concurrency_levels.append(concurrency)

        return concurrency_levels

    def calculate_statistics(self):
        elapsed = datetime.now() - self.start_time
        if not self.times:
            return None

        concurrency_levels = self.calculate_concurrency(
            self.start_end_times, self.args.duration, self.start_time
        )

        statistics_dict = {
            "target": self.args.target,
            "mode": self.args.unit,
            "rate": self.rate,
            "duration": self.args.duration,
            "input_tokens": self.n_input_tokens,
            "output_tokens": self.n_output_tokens,
            "mode_concurrency": stats.mode(concurrency_levels),
            "mean_concurrency": stats.mean(concurrency_levels),
            "median_concurrency": stats.median(concurrency_levels),
            "max_concurrency": max(concurrency_levels),
            "min_concurrency": min(concurrency_levels),
            "sstps_std": stats.stdev(self.sstps) if len(self.sstps) > 1 else None,
            "sstps_median": stats.median(self.sstps) if self.sstps else None,
            "sstps_mean": stats.mean(self.sstps) if self.sstps else None,
            "sstps_max": max(self.sstps) if self.sstps else None,
            "sstps_min": min(self.sstps) if self.sstps else None,
            "ttft_mean": (
                stats.mean(self.time_to_first_token)
                if self.args.target not in ["cog-triton", "triton"]
                and self.time_to_first_token
                else None
            ),
            "ttft_median": (
                stats.median(self.time_to_first_token)
                if self.args.target not in ["cog-triton", "triton"]
                and self.time_to_first_token
                else None
            ),
            "ttft_max": (
                max(self.time_to_first_token)
                if self.args.target not in ["cog-triton", "triton"]
                and self.time_to_first_token
                else None
            ),
            "ttft_min": (
                min(self.time_to_first_token)
                if self.args.target not in ["cog-triton", "triton"]
                and self.time_to_first_token
                else None
            ),
            "ttft_std": (
                stats.stdev(self.time_to_first_token)
                if self.args.target not in ["cog-triton", "triton"]
                and len(self.time_to_first_token) > 1
                else None
            ),
            "median_latency": round(stats.median(self.times), 3),
            "mean_latency": round(stats.mean(self.times), 3),
            "latency_std": stats.stdev(self.times) if len(self.times) > 1 else None,
            "max_latency": round(max(self.times), 3),
            "min_latency": round(min(self.times), 3),
            "server_actual_tps_mean": (
                stats.mean(self.server_side_actual_tps)
                if len(self.server_side_actual_tps) > 0
                else None
            ),
            "server_actual_tps_std": (
                stats.stdev(self.server_side_actual_tps)
                if len(self.server_side_actual_tps) > 1
                else None
            ),
            "server_actual_tps_median": (
                stats.median(self.server_side_actual_tps)
                if len(self.server_side_actual_tps) > 0
                else None
            ),
            "server_actual_tps_min": (
                min(self.server_side_actual_tps)
                if len(self.server_side_actual_tps) > 0
                else None
            ),
            "server_actual_tps_max": (
                max(self.server_side_actual_tps)
                if len(self.server_side_actual_tps) > 0
                else None
            ),
            "server_actual_exec_time_mean": (
                stats.mean(self.server_side_actual_execution_time)
                if len(self.server_side_actual_execution_time) > 0
                else None
            ),
            "server_actual_exec_time_std": (
                stats.stdev(self.server_side_actual_execution_time)
                if len(self.server_side_actual_execution_time) > 1
                else None
            ),
            "server_actual_exec_time_median": (
                stats.median(self.server_side_actual_execution_time)
                if len(self.server_side_actual_execution_time) > 0
                else None
            ),
            "server_actual_exec_time_min": (
                min(self.server_side_actual_execution_time)
                if len(self.server_side_actual_execution_time) > 0
                else None
            ),
            "server_actual_exec_time_max": (
                max(self.server_side_actual_execution_time)
                if len(self.server_side_actual_execution_time) > 0
                else None
            ),
            "total_requests_made": self.n_requests_made,
            "total_requests_started": self.n_requests_started,
            "total_requests_completed": self.n_requests_completed,
            "failure_rate": (
                self.failures / self.n_requests_started
                if self.n_requests_started > 0
                else 0
            ),
            "total_failures": self.failures,
            "cog_already_running_prediction": (
                self.n_cog_already_running_prediction
                if "cog" in self.args.target
                else None
            ),
            "e2e_throughput": self.n_requests_completed / elapsed.total_seconds(),
        }
        # Write statistics_dict to a JSON file
        with open(f"{self.experiment_dir}/results.json", "w") as file:
            json.dump(statistics_dict, file)

        self.statistics_dict = statistics_dict

        return statistics_dict

    def print_statistics(self):
        if self.statistics_dict is None:
            print("No requests completed.")
            return
        statistics_dict = self.statistics_dict

        print("---" * 10)
        print("Test Configuration:")
        print("---" * 10)
        print(f"Target: {statistics_dict['target']}")
        print(f"Mode: {statistics_dict['mode']}")
        print(f"Rate: {statistics_dict['rate']} {statistics_dict['mode']}")
        print(f"Duration: {statistics_dict['duration']} seconds")
        print(f"Input tokens: {statistics_dict['input_tokens']}")
        print(f"Output tokens: {statistics_dict['output_tokens']}")
        print("---" * 10)
        print("Concurrency levels:")
        print(f"Mode concurrency: {statistics_dict['mode_concurrency']}")
        print(f"Mean concurrency: {statistics_dict['mean_concurrency']}")
        print(f"Median concurrency: {statistics_dict['median_concurrency']}")
        print(f"Max concurrency: {statistics_dict['max_concurrency']}")
        print(f"Min concurrency: {statistics_dict['min_concurrency']}")
        print("---" * 10)
        print("Statistics for completed predictions:")
        print("---" * 10)
        print("Response times (seconds):")
        print("---" * 10)
        if statistics_dict["sstps_mean"] is not None:
            print("Single-stream TPS:")
            if statistics_dict["sstps_std"] is not None:
                print(f"SSTPS - Std: {statistics_dict['sstps_std']:.3f}")
            print(f"SSTPS - Median: {statistics_dict['sstps_median']:.3f}")
            print(f"SSTPS - Mean: {statistics_dict['sstps_mean']:.3f}")
            print(f"SSTPS - Max: {statistics_dict['sstps_max']:.3f}")
            print(f"SSTPS - Min: {statistics_dict['sstps_min']:.3f}")
            print("---" * 10)

        if statistics_dict["ttft_mean"] is not None:
            print("Time-to-First-Token Statistics (seconds):")
            print(f"Mean: {statistics_dict['ttft_mean']:.3f}")
            print(f"Median: {statistics_dict['ttft_median']:.3f}")
            print(f"Max: {statistics_dict['ttft_max']:.3f}")
            print(f"Min: {statistics_dict['ttft_min']:.3f}")
            if statistics_dict["ttft_std"] is not None:
                print(f"Std: {statistics_dict['ttft_std']:.3f}")
            print("---" * 10)

        print("Median response latency:", statistics_dict["median_latency"], "seconds")
        print("Mean response latency:", statistics_dict["mean_latency"], "seconds")
        if statistics_dict["latency_std"] is not None:
            print(
                f"Response Latency - Std: {statistics_dict['latency_std']:.3f} seconds"
            )
        print("Max response latency:", statistics_dict["max_latency"], "seconds")
        print("Min response latency:", statistics_dict["min_latency"], "seconds")

        print("---" * 10)
        print("Server-side metrics:")
        print("---" * 10)
        if statistics_dict["server_actual_tps_mean"] is not None:
            print("Server-side TPS")
            print(f"--Actual mean: {statistics_dict['server_actual_tps_mean']:.3f}")
            if statistics_dict["server_actual_tps_std"] is not None:
                print(f"--Actual std: {statistics_dict['server_actual_tps_std']:.3f}")
            else:
                print("--Actual std: N/A")
            print(f"--Actual median: {statistics_dict['server_actual_tps_median']:.3f}")
            print(f"--Actual min: {statistics_dict['server_actual_tps_min']:.3f}")
            print(f"--Actual max: {statistics_dict['server_actual_tps_max']:.3f}")

            print("Response Latency")
            print(
                f"--Actual mean: {statistics_dict['server_actual_exec_time_mean']:.3f}"
            )
            if statistics_dict["server_actual_exec_time_std"] is not None:
                print(
                    f"--Actual std: {statistics_dict['server_actual_exec_time_std']:.3f}"
                )
            else:
                print("--Actual std: N/A")
            print(
                f"--Actual median: {statistics_dict['server_actual_exec_time_median']:.3f}"
            )
            print(f"--Actual min: {statistics_dict['server_actual_exec_time_min']:.3f}")
            print(f"--Actual max: {statistics_dict['server_actual_exec_time_max']:.3f}")

        print("---" * 10)
        print(f"Total requests made: {statistics_dict['total_requests_made']}")
        print(f"Total requests started: {statistics_dict['total_requests_started']}")
        print(
            f"Total requests completed: {statistics_dict['total_requests_completed']}"
        )
        print(
            f"Failure rate: {statistics_dict['failure_rate']:.3f}, Total failures: {statistics_dict['total_failures']}"
        )
        if statistics_dict["cog_already_running_prediction"] is not None:
            print(
                f"Cog already running prediction: {statistics_dict['cog_already_running_prediction']}"
            )
        print(f"E2E throughput: {statistics_dict['e2e_throughput']:.3f} rps")


def load_experiment_results(experiment_dir):
    experiment_results = []

    # Iterate over each subdirectory in the experiment directory
    for sub_dir in os.listdir(experiment_dir):
        sub_dir_path = os.path.join(experiment_dir, sub_dir)

        # Check if the subdirectory contains a results.json file
        results_file_path = os.path.join(sub_dir_path, "results.json")
        if os.path.isfile(results_file_path):
            # Load the results.json file
            with open(results_file_path, "r") as file:
                results_data = json.load(file)
                experiment_results.append(results_data)

    return experiment_results


def generate_summary_plots(experiment_results, plot_dir):
    # Extract the data for plotting
    rates = []
    token_combinations = []
    sstps_values = {}
    sstps_errors = {}
    latency_values = {}
    latency_errors = {}
    ttft_values = {}
    ttft_errors = {}

    total_requests_made = 0
    duration = 0
    target = ""

    for result in experiment_results:
        rate = result["rate"]
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        token_combination = f"{input_tokens}:{output_tokens}"

        if rate not in rates:
            rates.append(rate)

        if token_combination not in token_combinations:
            token_combinations.append(token_combination)

        if token_combination not in sstps_values:
            sstps_values[token_combination] = {}
        sstps_values[token_combination][rate] = result["sstps_mean"]

        if token_combination not in sstps_errors:
            sstps_errors[token_combination] = {}
        sstps_errors[token_combination][rate] = result["sstps_std"] * 2

        if token_combination not in latency_values:
            latency_values[token_combination] = {}
        latency_values[token_combination][rate] = result["mean_latency"]

        if token_combination not in latency_errors:
            latency_errors[token_combination] = {}
        latency_errors[token_combination][rate] = result["latency_std"] * 2

        if token_combination not in ttft_values:
            ttft_values[token_combination] = {}
        ttft_values[token_combination][rate] = result["ttft_mean"]

        if token_combination not in ttft_errors:
            ttft_errors[token_combination] = {}
        ttft_errors[token_combination][rate] = result["ttft_std"] * 2

        total_requests_made += result["total_requests_made"]
        duration = result["duration"]
        target = result["target"][:6]

    # Sort the rates in ascending order
    rates.sort()

    # Create subplots for sstps, latency, and ttft
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Target: {target}, Duration: {duration}s, Total Requests: {total_requests_made}"
    )

    # Plot sstps
    x = np.arange(len(rates))
    width = 0.35 / len(token_combinations)
    for i, tc in enumerate(token_combinations):
        sstps_means = [sstps_values[tc].get(r, 0) for r in rates]
        sstps_errs = [sstps_errors[tc].get(r, 0) for r in rates]
        ax1.bar(x + i * width, sstps_means, width, yerr=sstps_errs, capsize=4, label=tc)
    ax1.set_ylabel("SSTPS Mean")
    ax1.set_title("SSTPS Mean by Rate and Token Combination")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rates)
    ax1.legend(title="Input Tokens:Output Tokens")

    # Plot latency
    for i, tc in enumerate(token_combinations):
        latency_means = [latency_values[tc].get(r, 0) for r in rates]
        latency_errs = [latency_errors[tc].get(r, 0) for r in rates]
        ax2.bar(
            x + i * width, latency_means, width, yerr=latency_errs, capsize=4, label=tc
        )
    ax2.set_ylabel("Mean Latency")
    ax2.set_title("Mean Latency by Rate and Token Combination")
    ax2.set_xticks(x)
    ax2.set_xticklabels(rates)
    ax2.legend(title="Input Tokens:Output Tokens")

    # Plot ttft
    for i, tc in enumerate(token_combinations):
        ttft_means = [ttft_values[tc].get(r, 0) for r in rates]
        ttft_errs = [ttft_errors[tc].get(r, 0) for r in rates]
        ax3.bar(x + i * width, ttft_means, width, yerr=ttft_errs, capsize=4, label=tc)
    ax3.set_ylabel("TTFT Mean")
    ax3.set_title("TTFT Mean by Rate and Token Combination")
    ax3.set_xticks(x)
    ax3.set_xticklabels(rates)
    ax3.legend(title="Input Tokens:Output Tokens", loc="upper left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    plot_filename = "experiment_plots.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"Plots saved to: {plot_path}")

    plt.close()


async def run_experiments(base_experiment_dir, original_args, n_io_tokens):
    rates = list(map(int, original_args.rate.split(",")))
    for rate, (n_input_tokens, n_output_tokens) in product(rates, n_io_tokens):
        experiment_dir = os.path.join(
            base_experiment_dir,
            f"rate_{rate}_n_io_tokens_{n_input_tokens}_{n_output_tokens}",
        )

        os.makedirs(experiment_dir, exist_ok=True)

        experiment = Experiment(
            experiment_dir, original_args, n_input_tokens, n_output_tokens, rate
        )

        await experiment.run()
        experiment.calculate_statistics()
        experiment.print_statistics()
        experiment.generate_plots()

    experiment_results = load_experiment_results(base_experiment_dir)
    generate_summary_plots(
        experiment_results=experiment_results, plot_dir=base_experiment_dir
    )


def parse_n_io_tokens(n_io_tokens_str):
    # Split the entire string by commas first to separate each pair
    pairs = n_io_tokens_str.split(",")

    # For each pair, split by the colon to separate input and output token lengths
    # and convert them to integers
    n_io_tokens = [tuple(map(int, pair.split(":"))) for pair in pairs]

    return n_io_tokens


def cli_main(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Benchmark script for Triton or Cog server."
    )
    parser.add_argument(
        "--target", required=True, help="Target server for the benchmark."
    )
    parser.add_argument(
        "--rate",
        required=True,
        type=str,
        help="Comma-separated list of rates (number of requests per second for 'rps' or total concurrent requests for 'batch'). Example: --rate 1,8,16",
    )
    parser.add_argument(
        "--n_io_tokens",
        required=True,
        type=str,
        help="Comma-separated list of pairs of input:output token lengths. Example: --n_io_tokens 128:128,512:128",
    )
    parser.add_argument(
        "--unit",
        type=str,
        choices=["rps", "batch"],
        required=True,
        help="Mode of operation: rps for requests per second, batch for concurrent requests.",
    )
    parser.add_argument(
        "--duration", type=int, required=True, help="Duration of test in seconds."
    )
    parser.add_argument(
        "--n_input_tokens", type=int, required=False, help="Number of input tokens."
    )
    parser.add_argument(
        "--n_output_tokens", type=int, required=False, help="Number of output tokens."
    )
    parser.add_argument(
        "--target_type",
        type=str,
        required=False,
        default=None,
        help="If target is a Replicate model, should be one of 'model', 'deployment', or 'official-model'",
    )
    original_args = parser.parse_args(args=args)

    rates = list(map(int, original_args.rate.split(",")))
    n_io_tokens = parse_n_io_tokens(original_args.n_io_tokens)

    base_dir = "perf-results"
    os.makedirs(base_dir, exist_ok=True)

    base_dir = "perf-results"
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_for_dir = (
        original_args.target
        if original_args.target in ["cog-triton", "triton"]
        else original_args.target[0:7]
    )
    rates_str = "-".join(map(str, rates))

    unique_dir_name = f"{timestamp}-{target_for_dir}-{original_args.unit}-{rates_str}-{original_args.n_io_tokens}-{original_args.duration}"
    base_experiment_dir = os.path.join(base_dir, unique_dir_name)
    os.makedirs(base_experiment_dir, exist_ok=True)

    # Construct a new argparse.Namespace object for each configuration

    asyncio.run(
        run_experiments(base_experiment_dir, original_args, n_io_tokens)
    )  # Use asyncio.run here to start run_experiments coroutine


if __name__ == "__main__":
    cli_main()
