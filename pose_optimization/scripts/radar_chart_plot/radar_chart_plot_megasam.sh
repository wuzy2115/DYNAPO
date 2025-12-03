#!/bin/bash

# Define the lists of CSV files and their corresponding labels
CSV_FILES=(
    "exps_BA/lightspeed_megasam_wo_BA_full.csv"
    "exps_BA/lightspeed_megasam_vinalla_BA_full.csv"
    "exps_BA/lightspeed_megasam_BA_full_their_mask.csv"
    "exps_BA/lightspeed_megasam_BA_full.csv"
)

LABELS=(
    "MegaSam"
    "MegaSam + BA w/o mask"
    "MegaSam + BA w/ MegaSam mask"
    "MegaSam + BA w/ our mask"
)

# Output filename for the plot
OUTPUT_FILENAME="radar_chart_megasam.svg"

# Construct the command with all arguments
python3 tools/plot_radar_chart.py \
    --csv_files "${CSV_FILES[@]}" \
    --labels "${LABELS[@]}" \
    --output_filename "$OUTPUT_FILENAME"

echo "Radar chart generated: $OUTPUT_FILENAME"
