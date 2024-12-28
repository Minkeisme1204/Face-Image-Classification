import os
import cv2
import matplotlib.pyplot as plt
import yaml

with open('Training_src/configs/params.yaml', 'r') as cfg_file:
    params = yaml.safe_load(cfg_file)

target_path = params.get('DATASET').get('PATH').get('TARGET_PATH')
raw_path = params.get('DATASET').get('PATH').get('RAW_PATH')

def read_yolo_label_file(label_path):
    """
    Read YOLO label file and extract class_id, x_center, y_center, width, height for each object.
    
    Args:
    - label_path (str): Path to the YOLO label file (e.g., 00001722.txt)
    
    Returns:
    - list of tuples: Each tuple is (class_id, x_center, y_center, width, height)
    """
    yolo_labels = []
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return []
    
    with open(label_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, values)
            class_id = int(class_id)
            yolo_labels.append((class_id, x_center, y_center, width, height))
    return yolo_labels


def denormalize_yolo_labels(labels, image_width, image_height):
    """
    Convert normalized YOLO labels back to absolute pixel coordinates (x0, y0, x1, y1).
    
    Args:
    - labels (list of tuples): Each tuple is (class_id, x_center, y_center, width, height)
    - image_width (int): Width of the image
    - image_height (int): Height of the image
    
    Returns:
    - list of tuples: Each tuple is (class_id, x0, y0, x1, y1)
    """
    denormalized_labels = []
    for class_id, x_center, y_center, width, height in labels:
        x0 = int((x_center - width / 2) * image_width)
        y0 = int((y_center - height / 2) * image_height)
        x1 = int((x_center + width / 2) * image_width)
        y1 = int((y_center + height / 2) * image_height)
        denormalized_labels.append((class_id, x0, y0, x1, y1))
    return denormalized_labels


def visualize_bounding_boxes(image_path, labels):
    """
    Draw bounding boxes on the image.
    
    Args:
    - image_path (str): Path to the image file
    - labels (list of tuples): Each tuple is (class_id, x0, y0, x1, y1)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
    
    for label in labels:
        class_id, x0, y0, x1, y1 = label
        color = (255, 0, 0)  # Red color for bounding box
        thickness = 2  # Line thickness
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Image: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()


def process_images_and_labels(image_dir, label_dir):
    """
    Process images and corresponding YOLO labels.
    
    Args:
    - image_dir (str): Path to the image folder
    - label_dir (str): Path to the folder containing YOLO label files
    """
    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
            
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            image_height, image_width = img.shape[:2]
            
            # Read YOLO labels
            yolo_labels = read_yolo_label_file(label_path)
            
            # Denormalize YOLO labels
            denormalized_labels = denormalize_yolo_labels(yolo_labels, image_width, image_height)
            
            # Visualize bounding boxes
            visualize_bounding_boxes(image_path, denormalized_labels)


# Example usage
image_dir = '/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/test/images'  # Replace with your image folder
label_dir = '/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/test/labels'  # Replace with your label folder

process_images_and_labels(image_dir, label_dir)
