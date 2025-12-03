#!/bin/bash

# Define the lists of CSV files and their corresponding labels
CSV_FILES=(
    "exps_BA/lightspeed_anycam_wo_BA_full.csv"
    "exps_BA/lightspeed_anycam_vinalla_BA_full_lr_5e-6.csv"
    "exps_BA/lightspeed_anycam_BA_full_their_mask.csv"
    "exps_BA/lightspeed_anycam_BA_full.csv"
)

LABELS=(
    "anycam"
    "anycam + BA w/o mask"
    "anycam + BA w/ anycam mask"
    "anycam + BA w/ our mask"
)

# Output filename for the plot
OUTPUT_FILENAME="radar_chart_anycam.svg"

# Construct the command with all arguments
python3 tools/plot_radar_chart.py \
    --csv_files "${CSV_FILES[@]}" \
    --labels "${LABELS[@]}" \
    --output_filename "$OUTPUT_FILENAME"

echo "Radar chart generated: $OUTPUT_FILENAME"
