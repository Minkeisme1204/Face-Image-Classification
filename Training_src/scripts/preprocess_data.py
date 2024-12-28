import os
import shutil
import random
import yaml 
import pandas as pd

with open('Training_src/configs/params.yaml', 'r') as cfg_file:
    params = yaml.safe_load(cfg_file)

target_path = params.get('DATASET').get('PATH').get('TARGET_PATH')
raw_path = params.get('DATASET').get('PATH').get('RAW_PATH')

def split_csv_and_create_label_files(csv_path, image_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split CSV data into train, val, and test sets and generate YOLO-style label files.
    
    Args:
    - csv_path (str): Path to the CSV file containing labels.
    - image_dir (str): Directory where the images are stored.
    - output_dir (str): Directory to store train, val, and test folders.
    - train_ratio (float): Proportion of images to be used for training.
    - val_ratio (float): Proportion of images to be used for validation.
    - test_ratio (float): Proportion of images to be used for testing.
    - random_state (int): Random seed for reproducibility.
    """
    # Create output folders for train, val, and test splits
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path, header=None, names=['image_name', 'image_width', 'image_height', 'x0', 'y0', 'x1', 'y1'])

    # Get unique image names
    unique_images = df['image_name'].unique()
    
    # Shuffle and split the data
    random.seed(random_state)
    random.shuffle(unique_images)
    
    train_size = int(len(unique_images) * train_ratio)
    val_size = int(len(unique_images) * val_ratio)
    test_size = len(unique_images) - train_size - val_size

    train_images = unique_images[:train_size]
    val_images = unique_images[train_size:train_size + val_size]
    test_images = unique_images[train_size + val_size:]

    print(f"Total Images: {len(unique_images)}")
    print(f"Train Images: {len(train_images)}, Val Images: {len(val_images)}, Test Images: {len(test_images)}")
    
    # Helper function to create label files for YOLO format
    def create_yolo_label_file(image_name, split):
        """Create a YOLO-style .txt label file for each image."""
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(output_dir, split, 'labels', image_name.replace('.jpg', '.txt'))
        
        # Get all bounding boxes for the current image
        image_data = df[df['image_name'] == image_name]
        
        with open(label_path, 'w') as f:
            for _, row in image_data.iterrows():
                image_height = int(row['image_height'])
                image_width = int(row['image_width'])
                x0 = int(row['x0'])
                y0 = int(row['y0'])
                x1 = int(row['x1'])
                y1 = int(row['y1'])
                
                # Calculate YOLO format (x_center, y_center, width, height) and normalize
                x_center = ((x0 + x1) / 2) / image_width
                y_center = ((y0 + y1) / 2) / image_height
                width = (x1 - x0) / image_width
                height = (y1 - y0) / image_height

                # Set the class_id (for now we use 0, but you can modify this logic)
                class_id = 0

                # Write to YOLO format file (class_id, x_center, y_center, width, height)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Copy the image to the appropriate split folder
        dest_image_path = os.path.join(output_dir, split, 'images', image_name)
        if os.path.exists(image_path):
            shutil.copy(image_path, dest_image_path)
    
    # Create label files for each split
    for split, image_list in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
        print(f"Processing {split} set...")
        for image_name in image_list:
            create_yolo_label_file(image_name, split)
    
    print("Data split and YOLO label files created successfully!")


if __name__ == '__main__':
    # Example usage
    input_image_dir = raw_path + '/images'  # Path to the folder containing image files
    input_label_dir = raw_path + '/faces.csv'  # Path to the folder containing label files
    output_dir = target_path  # Path to the folder where train, val, and test sets will be saved

    
    split_csv_and_create_label_files(input_label_dir, input_image_dir, output_dir)