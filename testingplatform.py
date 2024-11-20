import os
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count

def convert_to_jpg(image_path):
    """
    Converts a .JPEG image to .jpg format.
    """
    try:
        img = Image.open(image_path)
        new_path = image_path.with_suffix('.jpg')
        img.save(new_path, "JPEG")
        img.close()
        # Optionally remove the original .JPEG file
        os.remove(image_path)
        print(f"Converted: {image_path} -> {new_path}")
    except Exception as e:
        print(f"Failed to convert {image_path}: {e}")

def process_folder(folder_path):
    """
    Finds all .JPEG files in the folder and its subfolders.
    """
    folder_path = Path(folder_path)
    return [file for file in folder_path.rglob("*.JPEG")]

def distribute_conversion(image_paths):
    """
    Uses multiprocessing to convert images in parallel.
    """
    # Determine the number of processes to use
    num_processes = min(cpu_count(), len(image_paths))
    print(f"Using {num_processes} processes to process {len(image_paths)} images.")
    
    # Use multiprocessing to process the files
    with Pool(processes=num_processes) as pool:
        pool.map(convert_to_jpg, image_paths)

if __name__ == "__main__":
    # Specify the folder containing .JPEG files
    folder_path = input("Enter the folder path containing .JPEG files: ").strip()

    if not os.path.isdir(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        exit(1)

    # Collect all .JPEG files
    print("Scanning for .JPEG files...")
    jpeg_files = process_folder(folder_path)

    if not jpeg_files:
        print("No .JPEG files found.")
    else:
        print(f"Found {len(jpeg_files)} .JPEG files.")
        # Convert files in a distributed manner
        distribute_conversion(jpeg_files)
        print("Conversion complete!")
