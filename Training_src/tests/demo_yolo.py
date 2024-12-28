import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout, Reshape, Activation, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

batch_size = 8

def yolo_v1_tensor(labels, S=7, B=2):
    """
    Convert label data to a YOLOv1 output tensor.
    """
    tensor = np.zeros((S, S, B * 5))
    for label in labels:
        class_id, x_center, y_center, width, height = label
        i = int(x_center * S)
        j = int(y_center * S)
        x_relative = x_center * S - i
        y_relative = y_center * S - j

        for b in range(B):
            box_start = b * 5
            if tensor[i, j, box_start + 4] == 0:
                tensor[i, j, box_start:box_start + 5] = [x_relative, y_relative, width, height, 1]
                break
    return tensor

def load_data(image_dir, label_dir, image_size=(128,128), S=7, B=2):
    images = []
    labels = []
    for img_file, lbl_file in zip(sorted(os.listdir(image_dir)), sorted(os.listdir(label_dir))):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, lbl_file)

        img = image.load_img(img_path, target_size=image_size)
        img = image.img_to_array(img) / 255.0
        images.append(img)
        
        label_data = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id, x_center, y_center, width, height = map(float, parts)
                label_data.append([int(class_id), x_center, y_center, width, height])
        
        label_tensor = yolo_v1_tensor(label_data, S, B)
        labels.append(label_tensor)
    
    return np.array(images), np.array(labels)

image_train, train_out = load_data(
    '/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/train/images', 
    '/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/train/labels'
)

image_val, val_out = load_data(
    '/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/val/images', 
    '/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/dataset/val/labels'
)


# Model definition
inputs = tf.keras.Input(shape=(128, 128, 3))
x = Conv2D(8, (3, 3), activation='relu', name='conv1', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
x = Conv2D(8, (3, 3), activation='relu', name='conv2', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool2', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), activation='relu', name='conv3', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
x = Conv2D(16, (3, 3), activation='relu', name='conv4', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
x = BatchNormalization()(x)
# x = Conv2D(24, (3, 3), activation='relu', name='conv5', padding='same')(x)
# x = Conv2D(24, (3, 3), activation='relu', name='conv6', padding='same')(x)
# x = MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
x = Conv2D(32, (3, 3), activation='relu', name='conv7', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', name='conv8', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool5')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(7 * 7 * 10)(x)
output = Reshape((-1, 7, 7, 10))(output)

box1_activations = Activation('sigmoid')(output[..., 0:2])  # Sigmoid for x, y to constrain to [0, 1]
box2_activations = Activation('sigmoid')(output[..., 5:7])

# w, h (4 values for 2 boxes)
wh1_activations = Lambda(lambda x: tf.exp(x))(output[..., 2:4])  # Exponentiate w, h to ensure positivity
wh2_activations = Lambda(lambda x: tf.exp(x))(output[..., 7:9])
# Confidence scores (2 values for 2 boxes)
confidence1_activations = Activation('sigmoid')(output[..., 4:5])  # Sigmoid for confidence
confidence2_activations = Activation('sigmoid')(output[..., 9:10])
# Concatenate them back together
final_output = Concatenate(axis=-1)([box1_activations, wh1_activations, confidence1_activations, box2_activations, wh2_activations, confidence2_activations])

minke_model = Model(inputs=inputs, outputs=final_output)
minke_model.summary()

def yolo_loss(y_true, y_pred, S=7, B=2, lambda_coord=5, lambda_noobj=0.5):
    """
    YOLO Loss function with Object Mask
    
    Args:
    y_true -- ground truth tensor of shape (batch_size, S, S, B * 5)
    y_pred -- predicted tensor of shape (batch_size, S, S, B * 5)
    S -- size of the grid (7x7 by default)
    B -- number of bounding boxes per grid cell (default: 2)
    lambda_coord -- weight for coordinate loss
    lambda_noobj -- weight for confidence loss for cells without objects
    
    Returns:
    total_loss -- the total YOLO loss
    """
    # Get the object mask for each of the 2 bounding boxes
    object_mask1 = y_true[..., 4:5]  # Confidence for box 1
    object_mask2 = y_true[..., 9:10]  # Confidence for box 2
    
    # Coordinate loss (only apply when the object is present)
    xy1_loss = tf.reduce_sum(object_mask1 * tf.square(y_true[..., 0:2] - y_pred[..., 0:2 ]))
    xy2_loss = tf.reduce_sum(object_mask2 * tf.square(y_true[..., 5:7] - y_pred[..., 5:7 ]))
    
    # Width and height loss (only apply when the object is present)
    wh1_loss = tf.reduce_sum(object_mask1 * tf.square(tf.sqrt(y_true[..., 2:4]) - tf.sqrt(y_pred[..., 2:4])))
    wh2_loss = tf.reduce_sum(object_mask2 * tf.square(tf.sqrt(y_true[..., 7:9]) - tf.sqrt(y_pred[..., 7:9])))
    
    # Confidence loss (separate loss for objects and no-object cells)
    confidence1_loss = tf.reduce_sum(object_mask1 * tf.square(y_true[..., 4:5] - y_pred[..., 4:5]))  # Object present
    confidence2_loss = tf.reduce_sum(object_mask2 * tf.square(y_true[..., 9:10] - y_pred[..., 9:10]))  # Object present
    
    # No-object confidence loss (weight λ_noobj is used for boxes with no objects)
    no_object_loss1 = tf.reduce_sum((1 - object_mask1) * tf.square(y_true[..., 4:5] - y_pred[..., 4:5]))
    no_object_loss2 = tf.reduce_sum((1 - object_mask2) * tf.square(y_true[..., 9:10] - y_pred[..., 9:10]))
    
    # Total loss components
    coord_loss = lambda_coord * (xy1_loss + wh1_loss + xy2_loss + wh2_loss)
    object_loss = confidence1_loss + confidence2_loss
    no_object_loss = lambda_noobj * (no_object_loss1 + no_object_loss2)
    
    # Sum all the loss components
    total_loss = coord_loss + object_loss + no_object_loss
    
    return total_loss/batch_size

class MeanAveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, iou_threshold=0.5, name='mAP', **kwargs):
        super(MeanAveragePrecision, self).__init__(name=name, **kwargs)
        self.iou_threshold = iou_threshold
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')
        self.total_positives = self.add_weight(name='total_positives', initializer='zeros')

    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.process_batch(y_true, y_pred)
    
    def process_batch(self, y_true, y_pred):
        """
        Processes a batch of predictions and ground truth to calculate true positives, false positives, and false negatives.
        
        Args:
            y_true (tf.Tensor): Shape (batch_size, S, S, B * 5), ground truth boxes.
            y_pred (tf.Tensor): Shape (batch_size, S, S, B * 5), predicted boxes.
        """
        true_boxes = self.extract_boxes(y_true)  # Shape: (batch_size, N, 4)
        pred_boxes = self.extract_boxes(y_pred)  # Shape: (batch_size, M, 4)
        
        batch_size = tf.shape(true_boxes)[0]
        
        def process_single_image(args):
            true_boxes, pred_boxes = args
            
            # Calculate IoU for all predicted boxes with all ground-truth boxes
            iou_matrix = self.calculate_iou(pred_boxes, true_boxes)  # Shape: (M, N)
            
            # Get the best IoU for each predicted box and the corresponding index of the ground-truth box
            max_iou = tf.reduce_max(iou_matrix, axis=1)  # Shape: (M,)
            best_gt_index = tf.argmax(iou_matrix, axis=1)  # Shape: (M,)
            
            # Count true positives, false positives, and false negatives
            true_positives_mask = (max_iou >= self.iou_threshold)  # Shape: (M,)
            false_positives_mask = tf.logical_not(true_positives_mask)
            
            num_true_positives = tf.reduce_sum(tf.cast(true_positives_mask, tf.int32))
            num_false_positives = tf.reduce_sum(tf.cast(false_positives_mask, tf.int32))
            
            # Calculate number of false negatives as the number of unmatched ground-truth boxes
            matched_gt_indices = tf.gather(best_gt_index, tf.where(true_positives_mask)[:, 0])
            matched_gt_indices = tf.unique(matched_gt_indices).y
            num_false_negatives = tf.shape(true_boxes)[0] - tf.shape(matched_gt_indices)[0]
            
            return num_true_positives, num_false_positives, num_false_negatives, tf.shape(true_boxes)[0]
        
        # Process each image in the batch using `tf.map_fn`
        results = tf.map_fn(
            process_single_image, 
            (true_boxes, pred_boxes), 
            dtype=(tf.int32, tf.int32, tf.int32, tf.int32)
        )

        # Unpack the results
        true_positives, false_positives, false_negatives, total_positives = results
        
        # Update the counters
        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))
        self.total_positives.assign_add(tf.reduce_sum(total_positives))

    def extract_boxes(self, y):
        """
        Extracts bounding boxes from the output tensor `y` of shape (batch_size, S, S, B * 5).
        
        Args:
            y (tf.Tensor): Tensor of shape (batch_size, S, S, B * 5) representing the output from a YOLO-like model.
        
        Returns:
            tf.Tensor: A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        """
        
        # Ensure the correct input shape
        if len(y.shape) == 5:  # Handle case where y has an extra batch dimension
            y = tf.squeeze(y, axis=1)  # Remove extra dimension
        
        # Get the batch size and size S from the shape of the tensor
        batch_size = tf.shape(y)[0]  # Batch size
        S = tf.shape(y)[1]  # Size of the grid (SxS)
        num_channels = tf.shape(y)[-1]  # Last dimension, which is B * 5
        B = num_channels // 5  # Calculate B dynamically
        
        # Reshape the tensor to (batch_size, S, S, B, 5) where each box has [x, y, w, h, conf]
        y_reshaped = tf.reshape(y, (batch_size, S, S, B, 5))
        
        # Extract the components from the reshaped tensor
        x = y_reshaped[..., 0]
        y = y_reshaped[..., 1]
        w = y_reshaped[..., 2]
        h = y_reshaped[..., 3]
        conf = y_reshaped[..., 4]
        
        # Create a mask for boxes with confidence > 0.5
        conf_mask = conf > 0.5  # Shape: (batch_size, S, S, B)
        
        # Calculate x_min, y_min, x_max, y_max for each box
        x_min = (tf.range(S, dtype=tf.float32)[None, :, None, None] + x - w / 2) / tf.cast(S, tf.float32)
        y_min = (tf.range(S, dtype=tf.float32)[None, None, :, None] + y - h / 2) / tf.cast(S, tf.float32)
        x_max = (tf.range(S, dtype=tf.float32)[None, :, None, None] + x + w / 2) / tf.cast(S, tf.float32)
        y_max = (tf.range(S, dtype=tf.float32)[None, None, :, None] + y + h / 2) / tf.cast(S, tf.float32)
        
        # Concatenate the box coordinates
        boxes = tf.stack([x_min, y_min, x_max, y_max], axis=-1)  # Shape: (batch_size, S, S, B, 4)
        
        # Use tf.boolean_mask with corrected shape
        boxes = tf.boolean_mask(boxes, conf_mask)  # Now mask will work
        return boxes

    def calculate_iou(self, pred_boxes, true_boxes):
        """
        Calculate IoU between predicted boxes and ground-truth boxes.
        
        Args:
            pred_boxes (tf.Tensor): Shape (N, 4) where N is the number of predicted boxes.
            true_boxes (tf.Tensor): Shape (M, 4) where M is the number of true boxes.
        
        Returns:
            iou (tf.Tensor): Shape (N, M) where each element is the IoU between a predicted box and a ground-truth box.
        """
        pred_boxes = tf.expand_dims(pred_boxes, axis=1)  # Shape: (N, 1, 4)
        true_boxes = tf.expand_dims(true_boxes, axis=0)  # Shape: (1, M, 4)

        x_min = tf.maximum(pred_boxes[..., 0], true_boxes[..., 0])
        y_min = tf.maximum(pred_boxes[..., 1], true_boxes[..., 1])
        x_max = tf.minimum(pred_boxes[..., 2], true_boxes[..., 2])
        y_max = tf.minimum(pred_boxes[..., 3], true_boxes[..., 3])

        intersection_area = tf.maximum(0.0, x_max - x_min) * tf.maximum(0.0, y_max - y_min)

        pred_box_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        true_box_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])

        union_area = pred_box_area + true_box_area - intersection_area

        iou = tf.where(union_area > 0, intersection_area / union_area, tf.zeros_like(union_area))

        return iou
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
        recall = self.true_positives / (self.total_positives + 1e-6)
        return precision * recall
    
    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)
        self.total_positives.assign(0.0)


# Compile the model
minke_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=lambda y_true, y_pred: yolo_loss(y_true, y_pred, S=7, B=2)
    
)


# Train the model
history = minke_model.fit(
    x=image_train, 
    y=train_out, 
    validation_data=(image_val, val_out), 
    epochs=200, 
    batch_size=batch_size,
    callbacks=[EarlyStopping(patience=50), TensorBoard(log_dir='./logs')]
    
)
