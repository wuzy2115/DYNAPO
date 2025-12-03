#!/bin/bash

# Define the lists of CSV files and their corresponding labels
CSV_FILES=(
    "exps_BA/lightspeed_cut3r_wo_BA_full.csv"
    "exps_BA/lightspeed_cut3r_vinalla_BA_full_lr_1e-5.csv"
    "skip"
    "exps_BA/lightspeed_cut3r_BA_full_lr_1e-5.csv"
)

LABELS=(
    "cut3r"
    "cut3r + BA w/o mask"
    "skip"
    "cut3r + BA w/ our mask"
)

# Output filename for the plot
OUTPUT_FILENAME="radar_chart_cut3r.svg"

# Construct the command with all arguments
python3 tools/plot_radar_chart.py \
    --csv_files "${CSV_FILES[@]}" \
    --labels "${LABELS[@]}" \
    --output_filename "$OUTPUT_FILENAME"

echo "Radar chart generated: $OUTPUT_FILENAME"
