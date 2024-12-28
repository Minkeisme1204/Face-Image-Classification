import cupy as cp 

class Yolov1Loss(object):
    def __init__(self, S=5, B=2, lambda_coord=5, lambda_noobj=0.5):
        self.S = S
        self.B = B
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def __call__(self, predict, target):
        """
        predictions: cp.ndarray of shape (batch_size, S, S, B*5 + C)
        targets: cp.ndarray of shape (batch_size, S, S, B*5 + C)
        """

        batch_size = predict.shape[0]

        # Extract predicted bounding box coordinates, objectiveness
        pred_box = predict[..., :self.B*5].reshape(batch_size, self.S, self.S, self.B, 5)
        target_box = target[..., :self.B*5].reshape(batch_size, self.S, self.S, self.B)

        # Extract information of whether having objects or not
        object_exist = target[..., 4]
        non_object = ~object_exist

        # Calculate Confidence Loss
        confidence_loss_obj = cp.sum(object_exist * (pred_box[..., 4] - target_box[..., 4])**2)
        confidence_loss_noobj = self.lambda_noobj * cp.sum(non_object * (pred_box[..., 4] - target_box[..., 4])**2)

        # Calculate Localization Loss
        localization_loss = self.lambda_coord * cp.sum(
            object_exist[..., None] * ((pred_box[..., 0:2] - target_box[..., 0:2])**2 + 
                                       (cp.sqrt(cp.clip(pred_box[..., 2:4], 1e-6, None)) - cp.sqrt(cp.clip(target_box[..., 2:4], 1e-6, None)))**2))
        
        total_loss = localization_loss + confidence_loss_noobj + confidence_loss_obj

        return total_loss / batch_size

    def derivative(y_true, y_pred, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        # Split predictions into individual components
        pred_boxes = y_pred[..., :B*5]
        true_boxes = y_true[..., :B*5]
        
        # Reshape bounding boxes
        pred_box_reshape = pred_boxes.reshape((-1, S, S, B, 5))
        true_box_reshape = true_boxes.reshape((-1, S, S, B, 5))
        
        # Extract individual box components
        pred_x = pred_box_reshape[..., 0]
        pred_y = pred_box_reshape[..., 1]
        pred_w = pred_box_reshape[..., 2]
        pred_h = pred_box_reshape[..., 3]
        pred_confidence = pred_box_reshape[..., 4]
        
        true_x = true_box_reshape[..., 0]
        true_y = true_box_reshape[..., 1]
        true_w = true_box_reshape[..., 2]
        true_h = true_box_reshape[..., 3]
        true_confidence = true_box_reshape[..., 4]
        
        # === Derivative of Localization Loss ===
        dx = 2 * lambda_coord * (pred_x - true_x)
        dy = 2 * lambda_coord * (pred_y - true_y)
        dw = lambda_coord * (cp.sqrt(pred_w) - cp.sqrt(true_w)) / cp.sqrt(pred_w)
        dh = lambda_coord * (cp.sqrt(pred_h) - cp.sqrt(true_h)) / cp.sqrt(pred_h)
        
        # === Derivative of Confidence Loss ===
        dC = 2 * (pred_confidence - true_confidence) + 2 * lambda_noobj * (1 - true_confidence) * pred_confidence
        
        # Reshape derivatives for combination
        d_boxes = cp.stack([dx, dy, dw, dh, dC], axis=-1).reshape((-1, S, S, B * 5))
        d_output = d_boxes
        
        return d_output
    

# Giả sử ta có 2 batch, 7x7 lưới, B=2 bounding boxes mỗi ô, và C=20 classes
# batch_size = 2
# S = 7
# B = 2
# C = 20

# # Tạo ground truth và dự đoán ngẫu nhiên
# y_true = cp.random.rand(batch_size, S, S, B * 5 )  # Ground truth (2, 7, 7, 30)
# y_pred = cp.random.rand(batch_size, S, S, B * 5 )  # Predictions (2, 7, 7, 30)

# print(f"Shape of y_true: {y_true.shape}")  # (2, 7, 7, 30)
# print(f"Shape of y_pred: {y_pred.shape}")  # (2, 7, 7, 30)

class CatergoricalLossEntropy(object):
    def __init__(self):
        pass
    
    def __call__(self, y_pred, y_true):
        self.y = y_true
        # Calculate categorical cross-entropy loss
        loss = -cp.sum(y_true * cp.log(cp.clip(y_pred, 1e-15, 1. - 1e-15))) / y_pred.shape[0]
        return loss

    def derivative(self, y_pred):
        # Calculate derivative of categorical cross-entropy loss
        d_loss = (y_pred - self.y) / y_pred.shape[0]
        return d_loss
    
        

        