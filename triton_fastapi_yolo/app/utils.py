import cv2
import numpy as np

def preprocess_image(image_bytes: bytes, target_size=(640, 640)):
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("cv2.imdecode failed to decode the image bytes.")
        
    # Store original shape
    orig_shape = img.shape[:2]

    # Resize and pad
    # Simple resize for this demo
    img_resized = cv2.resize(img, target_size)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1 and HWC to CHW
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension -> (1, 3, 640, 640)
    input_tensor = np.expand_dims(img_chw, axis=0)
    
    return input_tensor, orig_shape

def postprocess_output(output_tensor, orig_shape, conf_threshold=0.25, iou_threshold=0.45):
    # YOLOv8 output: (1, 84, 8400)
    predictions = np.squeeze(output_tensor).T # (8400, 84)
    
    # xc, yc, w, h, class_probs
    boxes = predictions[:, :4]
    scores_matrix = predictions[:, 4:]
    
    # Get max score and class ID for each anchor
    class_ids = np.argmax(scores_matrix, axis=1)
    scores = np.max(scores_matrix, axis=1)
    
    # Filter by confidence
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    # Convert xc, yc, w, h to xmin, ymin, w, h
    x = boxes[:, 0] - boxes[:, 2] / 2
    y = boxes[:, 1] - boxes[:, 3] / 2
    w = boxes[:, 2]
    h = boxes[:, 3]
    
    boxes_xywh = np.stack((x, y, w, h), axis=-1)
    
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    
    results = []
    if len(indices) > 0:
        # Scale boxes back to original image size
        sx = orig_shape[1] / 640.0
        sy = orig_shape[0] / 640.0
        
        for i in indices.flatten():
            bx, by, bw, bh = boxes_xywh[i]
            results.append({
                "class_id": int(class_ids[i]),
                "confidence": float(scores[i]),
                "bbox": [
                    float(bx * sx), 
                    float(by * sy), 
                    float((bx + bw) * sx), 
                    float((by + bh) * sy)
                ]
            })
            
    return results
