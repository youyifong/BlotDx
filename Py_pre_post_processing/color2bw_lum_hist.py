import cv2
import argparse
import sys
import numpy as np
import tifffile as tiff
from pathlib import Path



def process_recursive(input_dir, output_dir):
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input folder '{input_dir}' does not exist.")
        sys.exit(1)

    # Find all .png files recursively (case-insensitive)
    files = list(input_path.rglob('*.[pP][nN][gG]'))

    if not files:
        print(f"No .png files found in {input_dir} or its subdirectories.")
        return

    print(f"Found {len(files)} images. Processing...")

    for img_file in files:
        # Determine relative path to recreate folder structure
        relative_path = img_file.relative_to(input_path)
        target_file_path = output_path / relative_path
        
        # Ensure the subfolder exists in the output directory
        target_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Process image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Skipping {img_file.name}: Could not read file.")
            continue

        # Convert to Grayscale (Luminosity)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform Histogram Equalization
        # equalized = cv2.equalizeHist(gray)
        equalized = gray

        # Save result
        cv2.imwrite(str(target_file_path), equalized)
        print(f"Saved: {relative_path}")

    print("\nRecursive processing complete.")



def weighted_grayscale(chunk):
    """
    Applies luminosity weights to a 3-channel chunk (RGB order).
    Formula: 0.299R + 0.587G + 0.114B
    """
    # chunk is expected to be (Height, Width, 3)
    weights = np.array([0.299, 0.587, 0.114])
    # Dot product performs the weighted sum across the last axis
    gray = np.dot(chunk, weights).astype(np.uint8)
    # return cv2.equalizeHist(gray)
    return gray

def process_recursive_tiff(input_dir, output_dir):
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.exists():
        print(f"Error: Input folder '{input_dir}' does not exist.")
        sys.exit(1)

    # Search for .tif and .tiff files
    files = list(input_path.rglob('*.tif*'))

    if not files:
        print(f"No TIFF files found in {input_dir}.")
        return

    print(f"Found {len(files)} TIFF files. Processing...")

    for img_file in files:
        relative_path = img_file.relative_to(input_path)
        target_file_path = output_path / relative_path
        target_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load multi-channel TIFF
            img = tiff.imread(str(img_file))
            
            # Ensure it has 6 channels (expecting shape: [H, W, 6] or [6, H, W])
            if img.ndim == 3 and img.shape[0] == 6:
                img = np.transpose(img, (1, 2, 0)) # Move channels to the end
            
            if img.shape[-1] != 6:
                print(f"Skipping {img_file.name}: Expected 6 channels, found {img.shape[-1]}")
                continue

            # Split into two 3-channel groups
            group_1 = img[:, :, 0:3]
            group_2 = img[:, :, 3:6]

            # Transform each group to grayscale + equalize
            ch1_processed = weighted_grayscale(group_1)
            ch2_processed = weighted_grayscale(group_2)

            # Stack back into a 2-channel image (Height, Width, 2)
            final_output = np.stack([ch1_processed, ch2_processed], axis=0)

            # Save as TIFF (OpenCV imwrite doesn't support arbitrary multi-channel well)
            tiff.imwrite(str(target_file_path), final_output, photometric='minisblack')
            print(f"Saved 2-channel TIFF: {relative_path}")

        except Exception as e:
            print(f"Failed to process {img_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively convert PNGs to grayscale and equalize.")
    parser.add_argument("--input", help="Source directory")
    parser.add_argument("--output", help="Destination directory")
    parser.add_argument(
            "--tiff", 
            action="store_true", 
            help="6 channel tiff files as input"
        )
    args = parser.parse_args()

    if args.tiff:
      process_recursive_tiff(args.input, args.output)
    else:
      process_recursive(args.input, args.output)