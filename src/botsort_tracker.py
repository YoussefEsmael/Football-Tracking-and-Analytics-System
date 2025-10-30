"""
BoT-SORT Tracker Integration for YOLOv11
Uses Ultralytics native BoT-SORT with proper configuration
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import yaml

try:
    from ultralytics import YOLO
    from ultralytics.trackers import BOTSORT
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[BoTSORT] Ultralytics not available")


class BoTSORTTracker:
    """
    BoT-SORT tracker wrapper for YOLOv11
    Integrates with YOLO's built-in tracking
    """
    
    def __init__(self, tracker_config: Optional[str] = None, 
                 track_high_thresh: float = 0.6,
                 track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.7,
                 track_buffer: int = 120,
                 match_thresh: float = 0.8,
                 fuse_score: bool = True,
                 proximity_thresh: float = 0.5,
                 appearance_thresh: float = 0.25,
                 with_reid: bool = True):
        """
        Initialize BoT-SORT tracker
        
        Args:
            tracker_config: Path to botsort.yaml config file
            track_high_thresh: High threshold for track confidence
            track_low_thresh: Low threshold for track confidence
            new_track_thresh: Threshold for creating new tracks
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
            fuse_score: Fuse detection and tracking scores
            proximity_thresh: Proximity threshold for ReID
            appearance_thresh: Appearance similarity threshold
            with_reid: Enable ReID features
        """
        
        self.tracker_config = tracker_config
        self.with_reid = with_reid
        
        # BoT-SORT parameters
        self.params = {
            'track_high_thresh': track_high_thresh,
            'track_low_thresh': track_low_thresh,
            'new_track_thresh': new_track_thresh,
            'track_buffer': track_buffer,
            'match_thresh': match_thresh,
            'fuse_score': fuse_score,
            'proximity_thresh': proximity_thresh,
            'appearance_thresh': appearance_thresh,
            'with_reid': with_reid
        }
        
        # Load custom config if provided
        if tracker_config and Path(tracker_config).exists():
            self._load_config(tracker_config)
        
        # Track management
        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        print(f"[BoTSORT] Initialized with ReID: {with_reid}")
        print(f"[BoTSORT] track_high_thresh: {self.params['track_high_thresh']}")
        print(f"[BoTSORT] track_buffer: {self.params['track_buffer']}")
        print(f"[BoTSORT] match_thresh: {self.params['match_thresh']}")
    
    def _load_config(self, config_path: str):
        """Load BoT-SORT configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update parameters from config
            if 'tracker_type' in config:
                print(f"[BoTSORT] Using tracker type: {config['tracker_type']}")
            
            # Update tracking parameters
            for key in self.params.keys():
                if key in config:
                    self.params[key] = config[key]
                    print(f"[BoTSORT] Config override: {key} = {config[key]}")
        
        except Exception as e:
            print(f"[BoTSORT] Could not load config: {e}")
    
    def update_with_yolo(self, model_results) -> List[Dict]:
        """
        Update tracker using YOLO model results directly
        This uses YOLO's built-in tracking
        
        Args:
            model_results: Results from model.track() call
        
        Returns:
            List of tracked detections with standardized format
        """
        self.frame_count += 1
        tracks = []
        
        if not model_results or len(model_results) == 0:
            return tracks
        
        result = model_results[0]
        
        # Check if tracking data is available
        if not hasattr(result, 'boxes') or result.boxes is None:
            return tracks
        
        boxes = result.boxes
        
        for i in range(len(boxes)):
            box = boxes[i]
            
            # Get track ID if available
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
            
            # Get bbox
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            
            # Get confidence
            conf = float(box.conf[0])
            
            # Get class
            cls = int(box.cls[0])
            
            tracks.append({
                'track_id': track_id,
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class': cls
            })
        
        return tracks
    
    def get_config_dict(self) -> Dict:
        """Get tracker configuration as dictionary for YOLO"""
        return self.params.copy()
    
    def save_config(self, output_path: str):
        """Save current configuration to YAML file"""
        config = {
            'tracker_type': 'botsort',
            **self.params
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"[BoTSORT] Config saved to {output_path}")


def create_botsort_config(output_path: str = "configs/botsort.yaml",
                         track_high_thresh: float = 0.6,
                         track_low_thresh: float = 0.1,
                         new_track_thresh: float = 0.7,
                         track_buffer: int = 30,
                         match_thresh: float = 0.8,
                         proximity_thresh: float = 0.5,
                         appearance_thresh: float = 0.25,
                         with_reid: bool = True):
    """
    Create BoT-SORT configuration file compatible with YOLOv11
    Uses exact parameter names expected by ultralytics
    """
    
    config = {
        # Tracker type
        'tracker_type': 'botsort',
        
        # Detection thresholds
        'track_high_thresh': track_high_thresh,
        'track_low_thresh': track_low_thresh,
        'new_track_thresh': new_track_thresh,
        
        # Track management
        'track_buffer': track_buffer,
        'match_thresh': match_thresh,
        'fuse_score': True,
        
        # ReID parameters
        'proximity_thresh': proximity_thresh,
        'appearance_thresh': appearance_thresh,
        'with_reid': with_reid,
        'reid_model': 'auto',  # Use built-in ReID model
        
        # Camera Motion Compensation (MUST use gmc_method not cmc_method)
        'gmc_method': 'sparseOptFlow',  # Options: 'sparseOptFlow', 'orb', 'ecc', 'none'
        
        # Frame rate
        'frame_rate': 30
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"[BoTSORT] Created config at {output_path}")
    print(f"[BoTSORT] Settings:")
    print(f"  - tracker_type: botsort")
    print(f"  - track_high_thresh: {track_high_thresh}")
    print(f"  - track_buffer: {track_buffer}")
    print(f"  - match_thresh: {match_thresh}")
    print(f"  - with_reid: {with_reid}")
    print(f"  - gmc_method: sparseOptFlow")
    
    return str(output_path)


def create_botsort_config_50fps(output_path: str = "configs/botsort_50fps.yaml"):
    """
    Create optimized BoT-SORT config for 50 FPS video
    """
    return create_botsort_config(
        output_path=output_path,
        track_high_thresh=0.65,  # Slightly higher for stability
        track_low_thresh=0.15,
        new_track_thresh=0.75,
        track_buffer=60,  # Double for 50 FPS
        match_thresh=0.85,
        proximity_thresh=0.6,  # More lenient for fast movement
        appearance_thresh=0.3,
        with_reid=True  # Enable ReID for better tracking
    )


# Example usage and testing
if __name__ == "__main__":
    # Create default config
    config_path = create_botsort_config("configs/botsort_default.yaml")
    
    # Create 50 FPS optimized config
    config_50fps = create_botsort_config_50fps("configs/botsort_50fps.yaml")
    
    # Initialize tracker
    tracker = BoTSORTTracker(
        tracker_config=config_50fps,
        with_reid=True
    )
    
    print("\n✓ BoT-SORT tracker initialized successfully")