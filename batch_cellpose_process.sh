#!/bin/bash

# Batch CellPose Processing Script (No Preprocessing)
# Usage: ./batch_cellpose_process.sh input_directory [diameter] [cellprob_threshold]

set -x # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 input_directory [diameter] [cellprob_threshold]"
    echo "  input_directory: Directory containing folders with image files"
    echo "  diameter: Cell diameter in pixels (default: 100)"
    echo "  cellprob_threshold: Cell probability threshold (default: 0)"
    echo ""
    echo "Example: $0 /path/to/image_folders 8 -2"
    exit 1
}

# Function to check if command exists
check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "Error: $1 is not installed or not in PATH"
        exit 1
    fi
}

# Function to extract statistics from CellPose output
extract_stats() {
    local seg_file="$1"
    python3 <<EOF
import numpy as np
import sys
import os

seg_file = "$seg_file"
if not os.path.exists(seg_file):
    print("0,0")  # cell_count, median_area
    sys.exit(0)

try:
    # Load the _seg.npy file
    data = np.load(seg_file, allow_pickle=True).item()
    masks = data['masks']
    
    # Calculate statistics
    unique_labels = np.unique(masks)
    cell_count = len(unique_labels) - 1  # subtract background (label 0)
    
    if cell_count > 0:
        # Calculate area for each cell (excluding background)
        cell_areas = []
        for label in unique_labels[1:]:  # skip background (0)
            area = np.sum(masks == label)
            cell_areas.append(area)
        
        median_area = np.median(cell_areas)
        print(f"{cell_count},{median_area:.1f}")
    else:
        print("0,0")
        
except Exception as e:
    print("0,0")
    sys.exit(0)
EOF
}

# Function to process a single image
process_image() {
    local input_file="$1"
    
    # Get base filename without extension
    local base_name=$(basename "$input_file" | sed 's/\.[^.]*$//')
    local input_dir=$(dirname "$input_file")
    
    echo "Processing: $input_file"
    
    # Run CellPose directly on original image (suppress verbose output)
    local cellpose_cmd="python -m cellpose \
        --image_path $input_file \
        --diameter 100 \
        --use_gpu --pretrained_model /media/gio/E8D0-7B68/train/models/cpsam_20251015_121000"
    
    if eval $cellpose_cmd >/dev/null 2>&1; then
        # Extract statistics
        local seg_file="${input_dir}/${base_name}_seg.npy"
        local stats=$(extract_stats "$seg_file")
        
        # Clean up CellPose output files
        rm -f "${input_dir}/${base_name}_seg.npy"
        
        echo "$stats"
    else
        echo "0,0"
    fi
}

# Check command line arguments
if [ $# -lt 1 ]; then
    usage
fi

INPUT_DIR="$1"
DIAMETER=${2:-100}
CELLPROB_THRESHOLD=${3:-0}

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found"
    exit 1
fi

# Check required commands
echo "Checking dependencies..."
check_command "python"  # Python for CellPose
check_command "python3" # Python3 for statistics

# Check if CellPose is available
if ! python -c "import cellpose" 2>/dev/null; then
    echo "Error: CellPose is not installed"
    echo "Install with: pip install cellpose"
    exit 1
fi

# Check for Excel/CSV Python libraries
if ! python -c "import pandas" 2>/dev/null; then
    echo "Error: pandas is not installed"
    echo "Install with: pip install pandas openpyxl"
    exit 1
fi

echo "========================================="
echo "Batch CellPose Processing (No Preprocessing)"
echo "========================================="
echo "Input directory: $INPUT_DIR"
echo "Cell diameter: $DIAMETER pixels"
echo "Cell probability threshold: $CELLPROB_THRESHOLD"
echo ""

# Create results CSV file
RESULTS_FILE="${INPUT_DIR}/cellpose_results.csv"
echo "folder_name,image_name,cell_count,median_cell_area" > "$RESULTS_FILE"

# Find all image files in subdirectories
image_count=0
processed_count=0

# Process each subdirectory
for folder_path in "$INPUT_DIR"/*; do
    if [ -d "$folder_path" ]; then
        folder_name=$(basename "$folder_path")
        echo "Processing folder: $folder_name"
        
        # Find image files in this folder
        find "$folder_path" -type f \( -iname "*.tif" -o -iname "*.tiff" -o -iname "*.JPG" -o -iname "*.jpg" -o -iname "*.jpeg" \) | while read -r image_file; do
            image_name=$(basename "$image_file")
            ((image_count++))
            
            echo "  Processing image $image_count: $image_name"
            
            # Process the image and get results
            results=$(process_image "$image_file" "$DIAMETER" "$CELLPROB_THRESHOLD")
            cell_count=$(echo "$results" | cut -d',' -f1)
            median_area=$(echo "$results" | cut -d',' -f2)
            
            # Append results to CSV
            echo "$folder_name,$image_name,$cell_count,$median_area" >> "$RESULTS_FILE"
            ((processed_count++))
            
            echo "    → Cells: $cell_count, Median area: $median_area pixels"
        done
    fi
done

# Convert CSV to Excel
echo ""
echo "Converting results to Excel format..."
python3 <<EOF
import pandas as pd
import sys

try:
    # Read CSV
    df = pd.read_csv('$RESULTS_FILE')
    
    # Write to Excel
    excel_file = '${INPUT_DIR}/cellpose_results.xlsx'
    df.to_excel(excel_file, index=False, engine='openpyxl')
    
    print(f"Excel file created: {excel_file}")
    
    # Print summary statistics
    total_images = len(df)
    total_cells = df['cell_count'].sum()
    avg_cells_per_image = df['cell_count'].mean()
    
    print(f"")
    print(f"=== PROCESSING SUMMARY ===")
    print(f"Total images processed: {total_images}")
    print(f"Total cells detected: {total_cells}")
    print(f"Average cells per image: {avg_cells_per_image:.1f}")
    print(f"Results saved to: {excel_file}")
    print(f"==========================")
    
except ImportError:
    print("Warning: openpyxl not available, keeping CSV format only")
    print(f"Results saved to: $RESULTS_FILE")
except Exception as e:
    print(f"Error creating Excel file: {e}")
    print(f"Results available in CSV format: $RESULTS_FILE")
EOF

echo ""
echo "✓ Batch processing completed successfully!"
echo "Results file: $RESULTS_FILE"
echo "✓ Cleanup completed!"
