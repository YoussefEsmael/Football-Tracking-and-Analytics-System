"""
Comprehensive configuration for football tracking system
All thresholds and parameters documented and centralized
"""
from pathlib import Path


class Config:
    """Main configuration class with all parameters"""
    
    # ============================================================================
    # INPUT/OUTPUT
    # ============================================================================
    VIDEO_PATH = r"C:\Users\yossef\Desktop\football is life\football_tracking\Full Match UCL 25-26  Barcelona v Paris St Germain  Match Day 2 1080p FHD-WDTeam ahdaf-kooora.com - Trim - Trim - Trim - Trim.mp4"
    OUTPUT_FILENAME = "output_tracked.mp4"
    EXPORT_DIRECTORY = "analytics_export"
    
    # ============================================================================
    # DETECTION SETTINGS
    # ============================================================================
    YOLO_MODEL_PATH = r"C:\Users\yossef\Desktop\football is life\football_tracking\models\best (4).pt"
    CONFIDENCE_THRESHOLD = 0.35
    

    
    # ============================================================================
    # REID SETTINGS (BoT-SORT handles ReID internally)
    # ============================================================================
    # BoT-SORT has built-in ReID, no separate model needed
    # Global ID manager uses BoT-SORT track IDs directly
    
    # Spatial-temporal validation for Global ID mapping
    REID_SPATIAL_THRESHOLD = 250  # Max movement between frames (pixels)
    REID_TEMPORAL_WINDOW = 60  # Max frames to keep track memory

    
    
    # Auto-adjust for high FPS videos
    AUTO_ADJUST_FOR_FPS = True  # Automatically scale thresholds based on FPS
    
    # ============================================================================
    # BYTETRACK SETTINGS
    # ============================================================================
    BYTETRACK_TRACK_THRESH = 0.75  # High confidence threshold for tracking
    BYTETRACK_TRACK_BUFFER = 120  # Frames to keep lost tracks
    BYTETRACK_MATCH_THRESH = 0.85  # IoU threshold for matching
    
    # ============================================================================
    # TEAM CLASSIFICATION SETTINGS
    # ============================================================================
    TEAM_N_TEAMS = 2  # Number of teams (always 2)
    TEAM_WARMUP_FRAMES = 25  # Frames to collect before classification
    TEAM_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for team assignment
    TEAM_ASSIGNMENT_LOCK = 0.7  # Confidence threshold for locking assignment
    
    # Jersey extraction
    JERSEY_MIN_PIXELS = 50  # Minimum valid pixels for color extraction
    JERSEY_TORSO_TOP = 0.15  # Top of torso region (fraction of height)
    JERSEY_TORSO_BOTTOM = 0.55  # Bottom of torso region
    JERSEY_TORSO_LEFT = 0.2  # Left edge of torso region (fraction of width)
    JERSEY_TORSO_RIGHT = 0.8  # Right edge of torso region
    
    # ============================================================================
    # ANALYTICS SETTINGS
    # ============================================================================
    FIELD_LENGTH = 105.0  # meters (standard football field)
    FIELD_WIDTH = 68.0  # meters
    
    # Speed thresholds
    HIGH_SPEED_THRESHOLD = 15.0  # km/h - threshold for high intensity running
    SPRINT_THRESHOLD = 20.0  # km/h - threshold for sprints
    MAX_REALISTIC_SPEED = 45.0  # km/h - cap for unrealistic values
    
    # Possession detection
    BALL_PROXIMITY_THRESHOLD = 50  # pixels - max distance for possession
    POSSESSION_MIN_FRAMES = 5  # Minimum frames to count as possession window
    
    # Pressing detection
    PRESSING_RADIUS = 100  # pixels - radius for pressing detection
    PRESSING_MIN_OPPONENTS = 3  # Minimum opponents for pressing event
    
    # Field zones (for zone control analysis)
    FIELD_ZONES_X = 3  # Divide field into 3 horizontal zones
    FIELD_ZONES_Y = 3  # Divide field into 3 vertical zones
    
    # ============================================================================
    # VISUALIZATION SETTINGS
    # ============================================================================
    OUTPUT_FPS = 25
    
    # Ellipse rendering
    ELLIPSE_ALPHA = 0.25  # Transparency of player ellipses
    ELLIPSE_GLOW_LAYERS = 3  # Number of glow layers
    
    # Trail settings
    TRAIL_LENGTH = 25  # Number of positions to keep in trail
    TRAIL_MIN_LENGTH = 2  # Minimum trail length to draw
    
    # Badge settings
    BADGE_RADIUS_PLAYER = 12  # Player badge radius
    BADGE_RADIUS_GOALKEEPER = 14  # Goalkeeper badge radius
    
    # Colors (BGR format)
    COLOR_TEAM_A = (0, 0, 255)  # Red
    COLOR_TEAM_B = (0, 255, 255)  # Yellow
    COLOR_UNKNOWN = (150, 150, 150)  # Gray
    COLOR_GOALKEEPER = (0, 215, 255)  # Gold
    COLOR_REFEREE = (255, 255, 0)  # Cyan
    COLOR_BALL = (0, 255, 255)  # Yellow
    
    # ============================================================================
    # EXPORT SETTINGS
    # ============================================================================
    GENERATE_HEATMAPS = True
    GENERATE_REPORTS = True
    EXPORT_DATA = True
    
    # Export formats
    EXPORT_CSV = True
    EXPORT_JSON = True
    EXPORT_PLOTS = True
    
    # Plot settings
    PLOT_DPI = 150
    PLOT_STYLE = 'whitegrid'  # seaborn style
    
    # ============================================================================
    # PERFORMANCE SETTINGS
    # ============================================================================
    ENABLE_GPU = True
    BATCH_SIZE_FEATURES = 8  # Batch size for feature extraction
    CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N frames
    MEMORY_CLEANUP_INTERVAL = 100  # Clean inactive IDs every N frames
    
    # Feature storage optimization
    MAX_FEATURES_PER_PLAYER = 30  # Keep only recent features
    FEATURE_STORAGE_INTERVAL = 1  # Store features every N frames
    
    # Memory management
    ENABLE_MEMORY_OPTIMIZATION = True  # Use memory-efficient feature storage
    MAX_TRACKLET_HISTORY = 100  # Maximum frames to store per tracklet
    
    # ============================================================================
    # DEBUG/LOGGING SETTINGS
    # ============================================================================
    VERBOSE = False  # Print detailed debug information
    LOG_REID_MATCHES = True  # Log re-identification matches
    LOG_TEAM_ASSIGNMENTS = True  # Log team assignment changes
    LOG_STATISTICS_INTERVAL = 100  # Print statistics every N frames
    
    # ============================================================================
    # CLASS MAPPING
    # ============================================================================
    CLASS_MAP = {
        0: 'ball',
        1: 'goalkeeper',
        2: 'player',
        3: 'referee'
    }
    
    CLASS_ID_BALL = 0
    CLASS_ID_GOALKEEPER = 1
    CLASS_ID_PLAYER = 2
    CLASS_ID_REFEREE = 3
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    MIN_BBOX_SIZE = 5  # Minimum bbox width/height in pixels
    MIN_CROP_AREA = 100  # Minimum crop area in pixels
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters"""
        errors = []
        
        # Check paths exist if specified
        if cls.YOLO_MODEL_PATH and not Path(cls.YOLO_MODEL_PATH).exists():
            errors.append(f"YOLO model not found: {cls.YOLO_MODEL_PATH}")
        
        # Check thresholds are in valid ranges
        if not 0 < cls.CONFIDENCE_THRESHOLD < 1:
            errors.append(f"CONFIDENCE_THRESHOLD must be in (0, 1)")
        
        # BoT-SORT thresholds (if defined)
        if hasattr(cls, 'BYTETRACK_MATCH_THRESH'):
            if not 0 < cls.BYTETRACK_MATCH_THRESH < 1:
                errors.append(f"BYTETRACK_MATCH_THRESH must be in (0, 1)")
        
        # Check team settings
        if cls.TEAM_N_TEAMS != 2:
            errors.append(f"TEAM_N_TEAMS must be 2 (got {cls.TEAM_N_TEAMS})")
        
        # Check spatial threshold is positive
        if hasattr(cls, 'REID_SPATIAL_THRESHOLD'):
            if cls.REID_SPATIAL_THRESHOLD <= 0:
                errors.append(f"REID_SPATIAL_THRESHOLD must be > 0")
        
        # Check temporal window is positive
        if hasattr(cls, 'REID_TEMPORAL_WINDOW'):
            if cls.REID_TEMPORAL_WINDOW <= 0:
                errors.append(f"REID_TEMPORAL_WINDOW must be > 0")
        
        if errors:
            print("[Config] ❌ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("[Config] ✓ Configuration validated successfully")
        return True
    
    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("\n" + "="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        print(f"Detection Model: {cls.YOLO_MODEL_PATH}")
        print(f"Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")
        
        # BoT-SORT settings
        if hasattr(cls, 'BYTETRACK_TRACK_THRESH'):
            print(f"\nBoT-SORT Tracking:")
            print(f"  Track Threshold: {cls.BYTETRACK_TRACK_THRESH}")
            print(f"  Track Buffer: {cls.BYTETRACK_TRACK_BUFFER} frames")
            print(f"  Match Threshold: {cls.BYTETRACK_MATCH_THRESH}")
        
        # ID Management
        print(f"\nGlobal ID Management:")
        print(f"  Spatial Threshold: {cls.REID_SPATIAL_THRESHOLD}px")
        print(f"  Temporal Window: {cls.REID_TEMPORAL_WINDOW} frames")
        if hasattr(cls, 'AUTO_ADJUST_FOR_FPS'):
            print(f"  Auto FPS Adjust: {cls.AUTO_ADJUST_FOR_FPS}")
        
        # Team Classification
        print(f"\nTeam Classification:")
        print(f"  Teams: {cls.TEAM_N_TEAMS}")
        print(f"  Warmup Frames: {cls.TEAM_WARMUP_FRAMES}")
        print(f"  Confidence Threshold: {cls.TEAM_CONFIDENCE_THRESHOLD}")
        
        # Output
        print(f"\nOutput:")
        print(f"  Video: {cls.OUTPUT_FILENAME}")
        print(f"  Export Directory: {cls.EXPORT_DIRECTORY}")
        print("="*80 + "\n")


# Validate configuration on import
if __name__ != "__main__":
    Config.validate()