#!/bin/bash

# Define the lists of CSV files and their corresponding labels
CSV_FILES=(
    "exps_BA/lightspeed_easi3r_wo_BA_full.csv"
    "exps_BA/lightspeed_easi3r_vinalla_BA_full.csv"
    "exps_BA/lightspeed_easi3r_BA_full_their_mask.csv"
    "exps_BA/lightspeed_easi3r_BA_full.csv"
)

LABELS=(
    "easi3r"
    "easi3r + BA w/o mask"
    "easi3r + BA w/ easi3r mask"
    "easi3r + BA w/ our mask"
)

# Output filename for the plot
OUTPUT_FILENAME="radar_chart_easi3r.svg"

# Construct the command with all arguments
python3 tools/plot_radar_chart.py \
    --csv_files "${CSV_FILES[@]}" \
    --labels "${LABELS[@]}" \
    --output_filename "$OUTPUT_FILENAME"

echo "Radar chart generated: $OUTPUT_FILENAME"
