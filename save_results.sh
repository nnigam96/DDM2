#!/bin/bash

# Get the last command executed in the terminal
last_command=$(history | tail -n 2 | head -n 1 | cut -c 8-)

# Extract the filename from the command
filename=$(basename $last_command .py)

# Set the directory where the generated opt file is located
source_dir="/staging/nnigam/ddm2/checkpoint"

# Set the directory where the renamed opt file will be saved
target_dir="/staging/nnigam/ddm2/opt_files"

# Set the prefix for the renamed opt file
prefix="$filename"

# Generate the new filename by adding the prefix and date/time suffix to the original filename
timestamp=$(date +"%Y%m%d-%H%M%S")
new_filename="${prefix}_${timestamp}_$(basename $source_dir/*.opt)"

# Copy the opt file to the target directory with the new filename
cp $source_dir/*.opt $target_dir/$new_filename

new_filename="${prefix}_${timestamp}_$(basename $source_dir/*.pth)"

# Copy the opt file to the target directory with the new filename
cp $source_dir/*.pth $target_dir/$new_filename
