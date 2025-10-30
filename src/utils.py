"""
Centralized utility functions for the football tracking system
All modules use these standardized functions
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import json


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))


def validate_bbox(bbox: Optional[Tuple], frame_shape: Tuple[int, int], 
                 min_size: int = 5) -> Optional[Tuple[int, int, int, int]]:
    """
    Centralized bbox validation used by ALL modules
    
    Args:
        bbox: (x1, y1, x2, y2)
        frame_shape: (height, width) or (height, width, channels)
        min_size: minimum width/height in pixels
    
    Returns:
        Validated (x1, y1, x2, y2) or None if invalid
    """
    if bbox is None or len(bbox) != 4:
        return None
    
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # Clamp to frame bounds
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)
    
    # Check minimum size
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    
    return (x1, y1, x2, y2)


def safe_crop(frame: np.ndarray, bbox: Tuple, min_area: int = 100) -> Optional[np.ndarray]:
    """
    Safely extract crop from frame with validation
    
    Args:
        frame: Input frame
        bbox: (x1, y1, x2, y2)
        min_area: Minimum crop area in pixels
    
    Returns:
        Cropped image or None if invalid
    """
    validated_bbox = validate_bbox(bbox, frame.shape, min_size=5)
    if validated_bbox is None:
        return None
    
    x1, y1, x2, y2 = validated_bbox
    
    # Check area
    area = (x2 - x1) * (y2 - y1)
    if area < min_area:
        return None
    
    try:
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None
        return crop
    except Exception:
        return None


def get_bbox_center(bbox: Tuple) -> Tuple[float, float]:
    """Calculate center of bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_bbox_area(bbox: Tuple) -> float:
    """Calculate area of bounding box"""
    x1, y1, x2, y2 = bbox
    return abs((x2 - x1) * (y2 - y1))


def iou(bbox1: Tuple, bbox2: Tuple) -> float:
    """
    Calculate Intersection over Union
    
    Args:
        bbox1, bbox2: (x1, y1, x2, y2)
    
    Returns:
        IoU score [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return np.sqrt(dx * dx + dy * dy)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """L2 normalize feature vector"""
    if features is None:
        return None
    
    norm = np.linalg.norm(features)
    if norm < 1e-8:
        return features
    
    return features / norm


def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two feature vectors
    
    Returns:
        Similarity score [-1, 1]
    """
    if feat1 is None or feat2 is None:
        return -1.0
    
    # Handle dimension mismatch
    if len(feat1) != len(feat2):
        min_len = min(len(feat1), len(feat2))
        feat1 = feat1[:min_len]
        feat2 = feat2[:min_len]
    
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return -1.0
    
    return float(np.dot(feat1, feat2) / (norm1 * norm2))


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_json_safe(data: Dict[str, Any], filepath: str):
    """Save data to JSON with numpy array handling"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)


def load_json_safe(filepath: str) -> Dict[str, Any]:
    """Load JSON file safely"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def draw_text_with_shadow(frame: np.ndarray, text: str, pos: Tuple[int, int],
                         font_scale: float = 0.5, color: Tuple = (255, 255, 255),
                         thickness: int = 1, shadow_offset: int = 2) -> np.ndarray:
    """
    Draw text with drop shadow for better visibility
    
    Args:
        frame: Input frame
        text: Text to draw
        pos: (x, y) position
        font_scale: Text size
        color: Text color (B, G, R)
        thickness: Text thickness
        shadow_offset: Shadow offset in pixels
    
    Returns:
        Frame with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    
    # Shadow
    cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    
    # Text
    cv2.putText(frame, text, (x, y),
                font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


class FrameRateCounter:
    """Track processing frame rate"""
    def __init__(self, window_size: int = 30):
        self.times = []
        self.window_size = window_size
        self.last_time = None
    
    def tick(self):
        """Record frame time"""
        import time
        current = time.time()
        
        if self.last_time is not None:
            self.times.append(current - self.last_time)
            if len(self.times) > self.window_size:
                self.times.pop(0)
        
        self.last_time = current
    
    def get_fps(self) -> float:
        """Get average FPS"""
        if not self.times:
            return 0.0
        
        avg_time = sum(self.times) / len(self.times)
        if avg_time == 0:
            return 0.0
        
        return 1.0 / avg_time


class ProgressTracker:
    """Track and display processing progress"""
    def __init__(self, total: int, name: str = "Processing"):
        self.total = total
        self.current = 0
        self.name = name
        self.start_time = None
        
    def start(self):
        """Start tracking"""
        import time
        self.start_time = time.time()
        self.current = 0
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress information"""
        import time
        
        if self.total == 0:
            return {'percentage': 0, 'eta': 0}
        
        percentage = (self.current / self.total) * 100
        
        if self.start_time is None or self.current == 0:
            eta = 0
        else:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed
            remaining = self.total - self.current
            eta = remaining / rate if rate > 0 else 0
        
        return {
            'percentage': percentage,
            'current': self.current,
            'total': self.total,
            'eta_seconds': eta
        }
    
    def print_progress(self):
        """Print progress bar"""
        progress = self.get_progress()
        bar_length = 40
        filled = int(bar_length * progress['percentage'] / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        eta_min = int(progress['eta_seconds'] / 60)
        eta_sec = int(progress['eta_seconds'] % 60)
        
        print(f"\r{self.name}: {bar} {progress['percentage']:.1f}% "
              f"({progress['current']}/{progress['total']}) "
              f"ETA: {eta_min:02d}:{eta_sec:02d}", end='', flush=True)


def batch_process(items, batch_size: int, process_fn):
    """Process items in batches"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
    
    return results