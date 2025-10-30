"""
Detection module with standardized output format
Uses centralized utilities for validation
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
from utils import validate_bbox, clamp

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False
    print("[Detection] ultralytics YOLO not available – will use motion detector fallback")


class DetectionManager:
    """
    Manages object detection using YOLO or motion-based fallback
    Outputs standardized detection format used by all modules
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 confidence_threshold: float = 0.5, 
                 use_yolo: bool = True,
                 device: str = 'cuda'):
        self.confidence_threshold = confidence_threshold
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.device = device
        self.model = None
        self.bg_subtractor = None
        
        # Statistics
        self.total_detections = 0
        self.frame_count = 0
        
        if self.use_yolo and model_path:
            try:
                self.model = YOLO(model_path)
                self.model.to(device)
                print(f"[Detection] ✓ YOLO model loaded from {model_path}")
                print(f"[Detection]   Device: {device}")
            except Exception as e:
                print(f"[Detection] ✗ Failed to load YOLO model: {e}")
                print("[Detection]   Falling back to motion detector")
                self.use_yolo = False
                self._init_motion_detector()
        elif not self.use_yolo:
            self._init_motion_detector()
        else:
            print("[Detection] ⚠ No model path provided")
    
    def _init_motion_detector(self):
        """Initialize background subtractor for motion detection"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=50,
            detectShadows=True
        )
        print("[Detection] ✓ Motion detector initialized")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            List of detection dicts with standardized format:
            {
                'bbox': (x1, y1, x2, y2),
                'confidence': float,
                'class': int,  # 0=ball, 1=goalkeeper, 2=player, 3=referee
                'track_id': None  # Filled by tracker
            }
        """
        self.frame_count += 1
        
        if self.use_yolo and self.model:
            detections = self._detect_yolo(frame)
        else:
            detections = self._detect_motion(frame)
        
        self.total_detections += len(detections)
        return detections
    
    def _detect_yolo(self, frame: np.ndarray) -> List[Dict]:
        """YOLO detection"""
        detections = []
        
        try:
            results = self.model(frame, imgsz=640, verbose=False)
            
            if not results or len(results) == 0:
                return detections
            
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Extract confidence
                conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
                
                if conf < self.confidence_threshold:
                    continue
                
                # Extract bbox
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy[0])
                x1, y1, x2, y2 = map(float, xyxy[:4])
                
                # Extract class
                cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class': cls if cls in [0, 1, 2, 3] else 2,  # Default to player
                    'track_id': None
                })
        
        except Exception as e:
            print(f"[Detection] ✗ YOLO detection error: {e}")
        
        return detections
    
    def _detect_motion(self, frame: np.ndarray, min_area: int = 500) -> List[Dict]:
        """Motion-based detection fallback"""
        if self.bg_subtractor is None:
            self._init_motion_detector()
        
        detections = []
        
        try:
            # Apply background subtraction
            mask = self.bg_subtractor.apply(frame)
            
            # Morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (players are taller than wide)
                aspect_ratio = h / (w + 1e-6)
                if aspect_ratio < 0.5 or aspect_ratio > 5.0:
                    continue
                
                detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.6,  # Fixed confidence for motion detection
                    'class': 2,  # Assume all motion detections are players
                    'track_id': None
                })
        
        except Exception as e:
            print(f"[Detection] ✗ Motion detection error: {e}")
        
        return detections
    
    def sanitize_detections(self, detections: List[Dict], 
                           frame_shape: tuple) -> List[Dict]:
        """
        Sanitize detections using centralized validation
        
        Args:
            detections: List of raw detections
            frame_shape: (height, width) or (height, width, channels)
        
        Returns:
            List of validated detections
        """
        sanitized = []
        
        for det in detections:
            # Validate bbox
            validated_bbox = validate_bbox(det['bbox'], frame_shape)
            
            if validated_bbox is None:
                continue
            
            # Create sanitized detection
            sanitized.append({
                'bbox': validated_bbox,
                'confidence': float(det.get('confidence', 0.5)),
                'class': int(det.get('class', 2)),
                'track_id': det.get('track_id', None)
            })
        
        return sanitized
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        avg_per_frame = self.total_detections / self.frame_count if self.frame_count > 0 else 0
        
        return {
            'total_frames': self.frame_count,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': avg_per_frame,
            'detection_method': 'YOLO' if self.use_yolo else 'Motion',
            'confidence_threshold': self.confidence_threshold
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.total_detections = 0
        self.frame_count = 0