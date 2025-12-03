#!/bin/bash

# Define the lists of CSV files and their corresponding labels
CSV_FILES=(
    "exps_BA/lightspeed_monst3r_wo_BA_full.csv"
    "exps_BA/lightspeed_monst3r_vinalla_BA_full.csv"
    "exps_BA/lightspeed_monst3r_BA_full_their_mask_lr_1e-5.csv"
    "exps_BA/lightspeed_monst3r_BA_full_lr_1e-5.csv"
)

LABELS=(
    "monst3r"
    "monst3r + BA w/o mask"
    "monst3r + BA w/ monst3r mask"
    "monst3r + BA w/ our mask"
)

# Output filename for the plot
OUTPUT_FILENAME="radar_chart_monst3r.svg"

# Construct the command with all arguments
python3 tools/plot_radar_chart.py \
    --csv_files "${CSV_FILES[@]}" \
    --labels "${LABELS[@]}" \
    --output_filename "$OUTPUT_FILENAME"

echo "Radar chart generated: $OUTPUT_FILENAME"
