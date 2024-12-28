import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import metrics

def yolo_v1_tensor(labels, S=7, B=2):
    """
    Convert label data to a YOLOv1 output tensor.
    
    Args:
        labels (list): List of labels where each label is [class_id, x_center, y_center, width, height].
        S (int): The grid size (default 7x7).
        B (int): Number of bounding boxes per grid cell.
        C (int): Number of classes.
        
    Returns:
        np.ndarray: YOLOv1 tensor of shape (S, S, B * 5 + C).
    """
    tensor = np.zeros((S, S, B * 5))  # Shape = (S, S, 10 + C) if B=2, C=20
    
    for label in labels:
        class_id, x_center, y_center, width, height = label

        # Calculate which cell (i, j) the center of the box falls into
        i = int(x_center * S)  # Row index
        j = int(y_center * S)  # Column index

        # Calculate x, y relative to the top-left corner of the grid cell
        x_relative = x_center * S - i
        y_relative = y_center * S - j

        # If this grid cell is already detecting an object, use the second box
        for b in range(B):
            box_start = b * 5  # Box index offset
            if tensor[i, j, box_start + 4] == 0:  # If objectness score is 0, use this box
                tensor[i, j, box_start:box_start + 5] = [x_relative, y_relative, width, height, 1]  # x, y, w, h, confidence
                break

    return tensor

image_size = (128,128)

# Read the dataset
train_images_list = os.listdir('/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/train/images')
train_labels_list = os.listdir('/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/train/labels')

val_images_list = os.listdir('/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/val/images')
val_labels_list = os.listdir('/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/val/labels')

image_train = []
label_data = []

for i, j in zip(range(0, len(train_images_list)), range(0, len(train_labels_list))):
    full_image_path, full_labels_path = os.path.join('/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/train/images', train_images_list[i]), os.path.join('/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/train/labels', train_labels_list[j])
    img = image.load_img(full_image_path, target_size = image_size, color_mode='rgb')
    img = image.img_to_array(img)
    img = img/255
    image_train.append(img)

    with open(full_labels_path, 'r') as file:
        for line in file:
            # Split the line and convert to float
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            label_data.append([int(class_id), x_center, y_center, width, height])

train_out = yolo_v1_tensor(label_data)
train_out_tf = [tf.convert_to_tensor(tensor, dtype=tf.float32) for tensor in train_out]
# train: image_train, train_out_tf

image_val =[]
label_val = []

for i, j in zip(range(0, len(val_images_list)), range(0, len(val_labels_list))):
    full_image_path, full_labels_path = os.path.join('/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/val/images', val_images_list[i]), os.path.join('/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/val/labels', val_labels_list[j])
    img = image.load_img(full_image_path, target_size = image_size, color_mode='rgb')
    img = image.img_to_array(img)
    img = img/255
    image_val.append(img)

    with open(full_labels_path, 'r') as file:
        for line in file:
            # Split the line and convert to float
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            label_val.append([int(class_id), x_center, y_center, width, height])

val_out = yolo_v1_tensor(label_val)
val_out_tf = [tf.convert_to_tensor(tensor, dtype=tf.float32) for tensor in val_out]


# model training

x = Input(shape=(image_size[0], image_size[1], 3), name = 'input')
x = Conv2D(8, (5, 5), activation='relu', name='conv1', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), activation='relu', name='conv2', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), activation='relu', name='conv3', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', name='conv4', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
x = BatchNormalization()(x)
x = Conv2D(24, (3, 3), activation='relu', name='conv5', padding='same')(x)
x = Conv2D(24, (3, 3), activation='relu', name='conv6', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
x = Flatten()(x)
x = Dense(, activation='relu')(x)
y = Dense(7*7* 2*5)(x)
output = tf.reshape(x, (-1, 7, 7, 2 * 5))

box_activations = tf.sigmoid(output[..., 0:2 * 2])  # x, y -> sigmoid
wh_activations = tf.exp(output[..., 2 * 2:4 * 2])  # w, h -> exp
confidence_activations = tf.sigmoid(output[..., 4 * 2:5 * 2])  # confidence -> sigmoid

minke_model = Model(inputs=x, outputs=output)
minke_model.summary()
def yolo_loss(y_true, y_pred, S=7, B=2, lambda_coord=5, lambda_noobj=0.5):
    """
    Custom YOLOv1 loss function.
    
    Args:
    - y_true: Ground-truth tensor of shape (batch_size, S, S, B * 5 + C)
    - y_pred: Predicted output tensor of shape (batch_size, S, S, B * 5 + C)
    """
    # Split y_true and y_pred into x, y, w, h, confidence, and class probabilities
    pred_box = y_pred[..., :B * 5]  # (batch, S, S, B * 5)
    pred_class_probs = y_pred[..., B * 5:]  # (batch, S, S, C)
    
    true_box = y_true[..., :B * 5]  # (batch, S, S, B * 5)
    true_class_probs = y_true[..., B * 5:]  # (batch, S, S, C)
    
    # 1️⃣ Localization Loss
    xy_loss = tf.reduce_sum(tf.square(true_box[..., 0:2 * B] - pred_box[..., 0:2 * B]))
    wh_loss = tf.reduce_sum(tf.square(tf.sqrt(true_box[..., 2 * B:4 * B]) - tf.sqrt(pred_box[..., 2 * B:4 * B])))
    localization_loss = lambda_coord * (xy_loss + wh_loss)
    
    # 2️⃣ Confidence Loss
    true_confidence = true_box[..., 4 * B:5 * B]  # (batch, S, S, B)
    pred_confidence = pred_box[..., 4 * B:5 * B]  # (batch, S, S, B)
    obj_conf_loss = tf.reduce_sum(tf.square(true_confidence - pred_confidence))
    noobj_conf_loss = lambda_noobj * tf.reduce_sum(tf.square(0 - pred_confidence))
    confidence_loss = obj_conf_loss + noobj_conf_loss
    
    total_loss = localization_loss + confidence_loss
    return total_loss

minke_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=lambda y_true, y_pred: yolo_loss(y_true, y_pred, 2),
    metrics=[metrics.calculate_map]
)