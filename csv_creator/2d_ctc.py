import os
import csv
import re
from pathlib import Path

def extract_metadata(filename):
    # Extract class (text before "_FOV1")
    class_match = re.match(r'^(.+?)_FOV1', filename)
    class_name = class_match.group(1) if class_match else ""
    
    # Extract frame number (number before "_t" and after "NA_")
    frame_match = re.search(r'NA_(\d+)_t', filename)
    frame_number = int(frame_match.group(1)) if frame_match else 0
    
    return class_name, frame_number

def generate_csv(root_dir, split, output_csv):
    # CSV header with specified column names
    csv_data = [["path", "class", "imagescore"]]
    
    # Datasets in 2D directory
    datasets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for dataset in datasets:
        new_ctc_path = os.path.join(root_dir, dataset, "New_CTC")
        if not os.path.exists(new_ctc_path):
            continue
            
        # Process specified split (train or test)
        split_path = os.path.join(new_ctc_path, split)
        if not os.path.exists(split_path):
            continue
            
        # Process video folders (01, 02, etc.)
        video_folders = [f for f in os.listdir(split_path) 
                        if os.path.isdir(os.path.join(split_path, f)) 
                        and f.isdigit()]
        
        for video in video_folders:
            video_path = os.path.join(split_path, video)
            # Process image files
            for img_file in os.listdir(video_path):
                if img_file.endswith(".tif"):
                    img_path = os.path.join(video_path, img_file)
                    class_name, frame_number = extract_metadata(img_file)
                    
                    # Use absolute path
                    abs_path = os.path.abspath(img_path)
                    csv_data.append([abs_path, class_name, frame_number])
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
    
    print(f"CSV file generated: {output_csv}")

if __name__ == "__main__":
    root_dir = "/nfs/NASSharedDrive/EngineeringTeam/Datasets/2D"
    generate_csv(root_dir, "train", "2d_ctc_train_dataset.csv")
    generate_csv(root_dir, "test", "2d_ctc_test_dataset.csv")