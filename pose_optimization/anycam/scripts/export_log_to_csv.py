#!/usr/bin/env python3

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple


def parse_log(filepath: str) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Parse a log file produced by evaluation to extract per-sequence metrics and average metrics.

    Returns:
      - sequence_names: list of sequence identifiers in the order they appear
      - per_sequence: mapping of sequence_name -> { 'ape_mean': float, 'rte_mean': float, 'rre_mean': float }
      - avg_metrics: mapping { 'ape_mean': float, 'rte_mean': float, 'rre_mean': float }
    """
    results_re = re.compile(r"^Results:\s+(\S+)")
    metric_patterns = {
        "ape_mean": re.compile(r"^\s*ape_mean\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"),
        "rte_mean": re.compile(r"^\s*rte_mean\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"),
        "rre_mean": re.compile(r"^\s*rre_mean\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"),
    }

    avg_results_header_re = re.compile(r"^Avg Results:")

    sequence_names: List[str] = []
    per_sequence: Dict[str, Dict[str, float]] = {}
    avg_metrics: Dict[str, float] = {}

    current_key: Optional[str] = None  # sequence name or "__AVG__"

    def ensure_entry(key: str) -> None:
        if key == "__AVG__":
            return
        if key not in per_sequence:
            per_sequence[key] = {}

    def capture_metric(target: Dict[str, float], line: str) -> None:
        for metric_name, pattern in metric_patterns.items():
            m = pattern.match(line)
            if m:
                try:
                    target[metric_name] = float(m.group(1))
                except ValueError:
                    pass

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Detect start of a results block for a specific sequence
            m_res = results_re.match(line)
            if m_res:
                current_key = m_res.group(1)
                if current_key not in sequence_names:
                    sequence_names.append(current_key)
                ensure_entry(current_key)
                continue

            # Detect start of Avg Results block
            if avg_results_header_re.match(line):
                current_key = "__AVG__"
                continue

            # Capture metrics if we are within a sequence or avg block
            if current_key is not None:
                if current_key == "__AVG__":
                    capture_metric(avg_metrics, line)
                    # If we've collected all three, we can stop capturing avg
                    if all(k in avg_metrics for k in metric_patterns.keys()):
                        current_key = None
                else:
                    capture_metric(per_sequence[current_key], line)
                    # If we've collected all three metrics for this sequence, end this block
                    if all(k in per_sequence[current_key] for k in metric_patterns.keys()):
                        current_key = None

    return sequence_names, per_sequence, avg_metrics


def write_four_row_csv(
    output_csv: str,
    sequence_names: List[str],
    per_sequence: Dict[str, Dict[str, float]],
    avg_metrics: Dict[str, float],
    user_avg_metrics: Optional[Dict[str, float]] = None,
) -> None:
    names_row: List[str] = []
    ape_row: List[str] = []
    rte_row: List[str] = []
    rre_row: List[str] = []

    # Sort sequence names before writing
    sorted_names = sorted(sequence_names)

    for name in sorted_names:
        names_row.append(name)
        seq_metrics = per_sequence.get(name, {})
        ape_row.append(str(seq_metrics.get("ape_mean", "")))
        rte_row.append(str(seq_metrics.get("rte_mean", "")))
        rre_row.append(str(seq_metrics.get("rre_mean", "")))

    # Append the Avg Results as the last column
    names_row.append("Avg Results")
    ape_row.append(str(avg_metrics.get("ape_mean", "")))
    rte_row.append(str(avg_metrics.get("rte_mean", "")))
    rre_row.append(str(avg_metrics.get("rre_mean", "")))

    if user_avg_metrics:
        names_row.append("User Avg Results")
        ape_row.append(str(user_avg_metrics.get("ape_mean", "")))
        rte_row.append(str(user_avg_metrics.get("rte_mean", "")))
        rre_row.append(str(user_avg_metrics.get("rre_mean", "")))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(names_row)
        writer.writerow(ape_row)
        writer.writerow(rte_row)
        writer.writerow(rre_row)


def calculate_user_avg_metrics(
    per_sequence: Dict[str, Dict[str, float]], user_sequences: List[str]
) -> Dict[str, float]:
    """Calculate avg metrics for a user-specified list of sequences."""
    ape_means: List[float] = []
    rte_means: List[float] = []
    rre_means: List[float] = []

    for seq_name in user_sequences:
        metrics = per_sequence.get(seq_name)
        if metrics:
            if "ape_mean" in metrics:
                ape_means.append(metrics["ape_mean"])
            if "rte_mean" in metrics:
                rte_means.append(metrics["rte_mean"])
            if "rre_mean" in metrics:
                rre_means.append(metrics["rre_mean"])

    return {
        "ape_mean": sum(ape_means) / len(ape_means) if ape_means else 0.0,
        "rte_mean": sum(rte_means) / len(rte_means) if rte_means else 0.0,
        "rre_mean": sum(rre_means) / len(rre_means) if rre_means else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse evaluation log and export a 4-row CSV.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the evaluation log file (e.g., lightspeed_megasam_wo_BA_full.txt)",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Path to output CSV file. Defaults to <input_basename>.csv in the same directory.",
    )
    parser.add_argument(
        "--sequences-dir",
        required=False,
        help="Path to the directory with all sequence subdirectories. If provided, sequences from this directory will be included in the CSV even if not in the log.",
    )
    parser.add_argument(
        "--user-sequences",
        nargs="+",
        help="A list of sequence names to calculate a separate average for.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{base}.csv")

    sequence_names, per_sequence, avg_metrics = parse_log(input_path)

    all_sequence_names = set(sequence_names)
    if args.sequences_dir:
        sequences_path = os.path.abspath(args.sequences_dir)
        if os.path.isdir(sequences_path):
            disk_sequences = [
                name
                for name in os.listdir(sequences_path)
                if os.path.isdir(os.path.join(sequences_path, name))
            ]
            all_sequence_names.update(disk_sequences)

    final_sequence_names = list(all_sequence_names)

    if not final_sequence_names:
        raise RuntimeError(
            "No sequences found in the log file or --sequences-dir. Ensure the input is correct."
        )

    user_avg_metrics = None
    if args.user_sequences:
        user_avg_metrics = calculate_user_avg_metrics(per_sequence, args.user_sequences)

    write_four_row_csv(
        output_path, final_sequence_names, per_sequence, avg_metrics, user_avg_metrics
    )
    print(f"Wrote CSV to: {output_path}")


if __name__ == "__main__":
    main()


