import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Load the pre-trained model
MODEL_DIR = 'ssd_mobilenet_v2_fpnlite_320x320/saved_model'
detection_model = tf.saved_model.load(MODEL_DIR)

# Load the label map
LABEL_MAP_PATH = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    detections = detection_model(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    return detections

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for the laptop webcam or provide the path to the video file
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detect_objects(frame)
        
        viz_utils.visualize_boxes_and_labels_on_image_array(
            frame,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
