import json
import cv2
import numpy as np
from typing import Dict, List, Any
from ultralytics import YOLO

class ObjectTracker:
    """Object tracking between frames (only conf > 0.65)"""
    
    def __init__(self, model_size: str = 'n'):
        self.detected_objects = {}
        self.next_object_id = 0
        self.confidence_threshold = 0.65  # confidence threshold
        
        # Load YOLO model
        model_path = f'yolov8{model_size}.pt'
        print(f"Loading YOLO model: {model_path}")
        self.yolo_model = YOLO(model_path)
        
        print(f"Saving objects with confidence > {self.confidence_threshold}")
    
    def detect_objects_2d(self, frame_np: np.ndarray) -> List[Dict]:
        """Object detection in frame with confidence filtering"""
        try:
            results = self.yolo_model(frame_np, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.yolo_model.names[class_id]
                        
                        # FILTERING: only objects with confidence > 0.65
                        if confidence > self.confidence_threshold:
                            detections.append({
                                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                'class_name': class_name,
                                'confidence': confidence,
                                'class_id': class_id
                            })
            
            print(f"Detected objects (conf > 0.65): {len(detections)}")
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []    

    def update_tracker(self, detections: List[Dict], frame_number: int) -> Dict:
        """Update tracker and return objects for current frame"""
        current_frame_objects = {}
        
        for detection in detections:
            object_id = self._assign_object_id(detection)
            
            if object_id not in self.detected_objects:
                self.detected_objects[object_id] = {
                    'object_id': object_id,
                    'class_name': detection['class_name'],
                    'first_seen': frame_number,
                    'detection_count': 1,
                    'max_confidence': detection['confidence'],
                    'last_seen': frame_number,
                    'all_detections': [detection]
                }
            else:
                self.detected_objects[object_id]['detection_count'] += 1
                self.detected_objects[object_id]['last_seen'] = frame_number
                self.detected_objects[object_id]['all_detections'].append(detection)
                self.detected_objects[object_id]['max_confidence'] = max(
                    self.detected_objects[object_id]['max_confidence'], 
                    detection['confidence']
                )
            
            current_frame_objects[object_id] = self.detected_objects[object_id]
            current_frame_objects[object_id]['current_detection'] = detection
        
        return current_frame_objects
    
    def _assign_object_id(self, detection: Dict) -> int:
        """Assign ID to object (simplified version without complex tracking)"""
        # Simple logic: each class gets its own IDs
        class_name = detection['class_name']
        
        # Look for existing object of same class in nearby position
        for obj_id, obj_data in self.detected_objects.items():
            if (obj_data['class_name'] == class_name and 
                obj_data['last_seen'] >= len(self.detected_objects) - 10):  # Recently seen
                return obj_id
        
        # New object
        new_id = self.next_object_id
        self.next_object_id += 1
        return new_id
    
    def get_summary_report(self) -> Dict:
        """Generate summary report"""
        summary = {}
        for obj_id, obj_data in self.detected_objects.items():
            class_name = obj_data['class_name']
            if class_name not in summary:
                summary[class_name] = []
            
            summary[class_name].append({
                'object_id': obj_id,
                'detection_count': obj_data['detection_count'],
                'max_confidence': obj_data['max_confidence'],
                'first_seen_frame': obj_data['first_seen'],
                'last_seen_frame': obj_data['last_seen']
            })
        
        return summary

def draw_detections(frame_np: np.ndarray, current_objects: Dict) -> np.ndarray:
    """Draw bbox only for objects with confidence > 0.65"""
    annotated_frame = frame_np.copy()
    
    # Fixed color set for stability
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]
    
    for obj_id, obj_data in current_objects.items():
        if 'current_detection' not in obj_data:
            continue
            
        detection = obj_data['current_detection']
        
        # CONFIDENCE CHECK: draw only if > 0.65
        if detection['confidence'] <= 0.65:
            continue
            
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Simple color selection by object ID
        color = colors[obj_id % len(colors)]
        
        # Draw bbox
        cv2.rectangle(annotated_frame, 
                     (int(x1), int(y1)), (int(x2), int(y2)), 
                     color, 2)
        
        # Label with class, ID and confidence
        label = f"{class_name} ID:{obj_id} ({confidence:.2f})"
        
        # Background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_frame, 
                     (int(x1), int(y1) - text_size[1] - 10),
                     (int(x1) + text_size[0], int(y1)),
                     color, -1)
        
        # Text
        cv2.putText(annotated_frame, label, 
                   (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return annotated_frame

def save_detection_report(tracker: ObjectTracker, video_path: str):
    """Save object detection report (conf > 0.65) to JSON file"""
    report = tracker.get_summary_report()
    report_path = video_path.replace('.mp4', '_detection_report.json')
    
    # Add meta-information to report
    full_report = {
        'metadata': {
            'total_unique_objects': sum(len(objects) for objects in report.values()),
            'total_classes': len(report),
            'confidence_threshold': 0.65,
            'video_source': video_path
        },
        'objects_by_class': report
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"Report saved: {report_path}")
    
    # Print statistics to console
    print("\n=== DETECTION STATISTICS (conf > 0.65) ===")
    total_objects = 0
    total_detections = 0
    
    for class_name, objects in report.items():
        class_detections = sum(obj['detection_count'] for obj in objects)
        total_detections += class_detections
        total_objects += len(objects)
        
        print(f"\n{class_name}:")
        print(f"  Unique objects: {len(objects)}")
        print(f"  Total detections: {class_detections}")
        
        for obj in objects:
            print(f"    ID {obj['object_id']}: {obj['detection_count']} detections, max confidence: {obj['max_confidence']:.3f}")
    
    print(f"\nTotal (conf > 0.65):")
    print(f"  Total unique objects: {total_objects}")
    print(f"  Total classes: {len(report)}")
    print(f"  Total detections: {total_detections}")