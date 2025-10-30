"""
Enhanced Football Tracking Pipeline - Complete Version
BoT-SORT Tracker (with built-in ReID) + SigLIP/UMAP Team Classification
No separate deep ReID extraction - BoT-SORT handles it internally
"""
import os
import sys
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

# Set OpenMP fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import configuration
from config import Config

# Import enhanced modules
from botsort_tracker import BoTSORTTracker, create_botsort_config_50fps, create_botsort_config
from siglip_team_classifier import SigLIPTeamClassifier
from id_manager_simple import SimpleGlobalIDManager

# Import utilities
from utils import validate_bbox, ProgressTracker, FrameRateCounter, save_json_safe

# Import existing modules
from visualizer import EllipticalVisualizer
from analytics import AdvancedAnalytics
from export_manager import ExportManager

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ERROR] ultralytics not available - install with: pip install ultralytics")


def run_enhanced_pipeline(video_path, output_path=None, max_frames=None,
                          tracker_config=None, use_siglip=True):
    """
    Enhanced pipeline with BoT-SORT + SigLIP
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        max_frames: Maximum frames to process (None = all)
        tracker_config: Path to BoT-SORT config YAML
        use_siglip: Use SigLIP for team classification
    """
    
    print("=" * 80)
    print("ENHANCED FOOTBALL TRACKING PIPELINE")
    print("=" * 80)
    print("Architecture:")
    print("  ✓ YOLOv11 Detection")
    print("  ✓ BoT-SORT Tracking (with built-in ReID)")
    print(f"  ✓ {'SigLIP + UMAP' if use_siglip else 'Color-based'} Team Classification")
    print("  ✓ Simple Global ID Management")
    print("  ✓ Advanced Analytics & Export")
    print("=" * 80)
    
    if not YOLO_AVAILABLE:
        raise RuntimeError("ultralytics YOLO not available!")
    
    Config.validate()
    Config.print_summary()
    
    # ========== VIDEO SETUP ==========
    
    print("\n" + "="*80)
    print("VIDEO SETUP")
    print("="*80)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Video: {Path(video_path).name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {src_fps:.2f}")
    print(f"Total frames: {total_frames}")
    
    # Auto-adjust thresholds for FPS
    auto_adjust = getattr(Config, 'AUTO_ADJUST_FOR_FPS', True)
    if auto_adjust and src_fps > 40:
        fps_ratio = src_fps / 25.0
        Config.REID_SPATIAL_THRESHOLD = int(200 * fps_ratio)
        Config.REID_TEMPORAL_WINDOW = int(90 * fps_ratio)
        Config.TEAM_WARMUP_FRAMES = int(100 * fps_ratio)
        
        print(f"\n⚠ High FPS ({src_fps:.1f}) detected - auto-adjusting:")
        print(f"  SPATIAL_THRESHOLD: {Config.REID_SPATIAL_THRESHOLD}px")
        print(f"  TEMPORAL_WINDOW: {Config.REID_TEMPORAL_WINDOW} frames")
        print(f"  WARMUP_FRAMES: {Config.TEAM_WARMUP_FRAMES} frames")
    
    # ========== INITIALIZE MODULES ==========
    
    print("\n" + "="*80)
    print("INITIALIZING MODULES")
    print("="*80)
    
    # 1. Create BoT-SORT config
    print("\n[1/5] Creating BoT-SORT Configuration...")
    if tracker_config is None:
        if src_fps > 40:
            tracker_config = create_botsort_config_50fps("configs/botsort_auto.yaml")
        else:
            tracker_config = create_botsort_config(
                "configs/botsort_auto.yaml",
                track_buffer=int(30 * (src_fps / 25)),
                with_reid=True
            )
    print(f"✓ Tracker config: {tracker_config}")
    
    # 2. Load YOLO model
    print("\n[2/5] Loading YOLO Model...")
    yolo_model = YOLO(Config.YOLO_MODEL_PATH)
    device = getattr(Config, 'REID_DEVICE', 'cuda')
    print(f"✓ Model: {Config.YOLO_MODEL_PATH}")
    print(f"  Device: {device}")
    
    # 3. Initialize Simple ID Manager (no features)
    print("\n[3/5] Initializing Global ID Manager...")
    id_manager = SimpleGlobalIDManager(
        spatial_threshold=Config.REID_SPATIAL_THRESHOLD,
        temporal_window=Config.REID_TEMPORAL_WINDOW
    )
    print("✓ ID Manager ready (BoT-SORT handles ReID)")
    
    # 4. Initialize Team Classifier
    print("\n[4/5] Initializing Team Classifier...")
    if use_siglip:
        try:
            team_classifier = SigLIPTeamClassifier(
                n_teams=Config.TEAM_N_TEAMS,
                warmup_frames=Config.TEAM_WARMUP_FRAMES,
                confidence_threshold=Config.TEAM_CONFIDENCE_THRESHOLD,
                model_name="google/siglip-base-patch16-224",
                device=device,
                use_umap=True,
                umap_n_components=8
            )
            print("✓ SigLIP + UMAP team classifier loaded")
        except Exception as e:
            print(f"✗ SigLIP failed: {e}")
            print("  Falling back to color-based classifier")
            from team_classifier import AdvancedTeamClassifier
            team_classifier = AdvancedTeamClassifier(
                n_teams=Config.TEAM_N_TEAMS,
                warmup_frames=Config.TEAM_WARMUP_FRAMES,
                confidence_threshold=Config.TEAM_CONFIDENCE_THRESHOLD,
                use_reid_features=False
            )
            use_siglip = False
    else:
        from team_classifier import AdvancedTeamClassifier
        team_classifier = AdvancedTeamClassifier(
            n_teams=Config.TEAM_N_TEAMS,
            warmup_frames=Config.TEAM_WARMUP_FRAMES,
            confidence_threshold=Config.TEAM_CONFIDENCE_THRESHOLD,
            use_reid_features=False
        )
        print("✓ Color-based team classifier loaded")
    
    # 5. Initialize Analytics & Visualization
    print("\n[5/5] Initializing Analytics & Visualization...")
    analytics = AdvancedAnalytics(
        field_dimensions=(Config.FIELD_LENGTH, Config.FIELD_WIDTH),
        fps=src_fps
    )
    analytics.calibrate_field(width, height)
    
    visualizer = EllipticalVisualizer(team_classifier=team_classifier)
    print("✓ Analytics and visualizer ready")
    
    # Setup output
    output_path = output_path or Config.OUTPUT_FILENAME
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Output] Writing to: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (width, height))
    
    if not writer.isOpened():
        print(f"[ERROR] Cannot open video writer for: {output_path}")
        print(f"[ERROR] Trying alternative codec...")
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_path}")
    
    # ========== PROCESSING ==========
    
    print("\n" + "="*80)
    print("PROCESSING VIDEO")
    print("="*80)
    print(f"Mode: BoT-SORT + {'SigLIP' if use_siglip else 'Color'}")
    print("="*80 + "\n")
    
    progress = ProgressTracker(total_frames, "Processing")
    fps_counter = FrameRateCounter()
    
    progress.start()
    frame_idx = 0
    start_time = time.time()
    
    team_assignments = {}
    
    # Reopen video
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break
        
        fps_counter.tick()
        
        # ========== YOLO + BoT-SORT TRACKING ==========
        # Use built-in tracker name instead of config file
        # Options: 'botsort.yaml', 'bytetrack.yaml'
        
        # Determine tracker to use
        if tracker_config and Path(tracker_config).exists():
            # Try to use provided config
            tracker_param = tracker_config
        else:
            # Use built-in BoT-SORT (more reliable)
            tracker_param = 'botsort.yaml'
            if frame_idx == 1:
                print(f"[Tracking] Using built-in tracker: {tracker_param}")
        
        try:
            results = yolo_model.track(
                frame,
                conf=Config.CONFIDENCE_THRESHOLD,
                iou=0.5,
                persist=True,
                tracker=tracker_param,
                verbose=False
            )
        except Exception as e:
            if frame_idx == 1:
                print(f"[Warning] Tracker error: {e}")
                print("[Tracking] Falling back to bytetrack")
            
            # Fallback: Try ByteTrack instead
            try:
                results = yolo_model.track(
                    frame,
                    conf=Config.CONFIDENCE_THRESHOLD,
                    iou=0.5,
                    persist=True,
                    tracker='bytetrack.yaml',
                    verbose=False
                )
            except:
                # Last resort: detection only
                if frame_idx == 1:
                    print("[Warning] All trackers failed, using detection only")
                results = yolo_model(frame, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
        
        # Extract tracks with IDs
        tracks = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    
                    # Get track ID from BoT-SORT
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                    else:
                        continue  # Skip if no track ID
                    
                    # Get bbox
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = tuple(map(float, xyxy))
                    
                    # Validate bbox
                    validated_bbox = validate_bbox(bbox, frame.shape)
                    if validated_bbox is None:
                        continue
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    tracks.append({
                        'track_id': track_id,
                        'bbox': validated_bbox,
                        'confidence': conf,
                        'class': cls
                    })
        
        # ========== GLOBAL ID MAPPING ==========
        for track in tracks:
            if track['class'] not in [Config.CLASS_ID_PLAYER, Config.CLASS_ID_GOALKEEPER]:
                track['global_id'] = -1
                continue
            
            # Map BoT-SORT track_id to global_id
            global_id, _ = id_manager.get_or_create_global_id(
                track_id=track['track_id'],
                bbox=track['bbox'],
                frame_idx=frame_idx,
                class_id=track['class']
            )
            
            track['global_id'] = global_id
        
        # ========== TEAM CLASSIFICATION ==========
        detection_dict = {}
        
        for track in tracks:
            if track['class'] == Config.CLASS_ID_PLAYER:
                gid = track.get('global_id', -1)
                if gid != -1:
                    detection_dict[gid] = {
                        'bbox': track['bbox'],
                        'class': track['class'],
                        'confidence': track['confidence']
                    }
        
        # Update team classifier (SigLIP extracts features internally)
        if use_siglip:
            team_classifier.update(frame, detection_dict, frame_idx)
        else:
            # Color-based needs explicit feature extraction
            color_features_dict = {}
            for gid, det in detection_dict.items():
                color_feat = team_classifier.extract_jersey_features(frame, det['bbox'])
                if color_feat is not None:
                    color_features_dict[gid] = color_feat
            team_classifier.update(frame, detection_dict, frame_idx, color_features_dict)
        
        # Get team assignments
        for gid in detection_dict.keys():
            team_id = team_classifier.get_team_assignment(gid)
            team_assignments[gid] = team_id
        
        # ========== ANALYTICS ==========
        analytics_detections = {}
        for track in tracks:
            gid = track.get('global_id', -1)
            if gid != -1:
                analytics_detections[gid] = {
                    'bbox': track['bbox'],
                    'class': 'player' if track['class'] == Config.CLASS_ID_PLAYER else 'goalkeeper',
                    'confidence': track['confidence']
                }
        
        timestamp = frame_idx / src_fps
        analytics.update_positions(analytics_detections, team_assignments, frame_idx, timestamp)
        
        # ========== VISUALIZATION ==========
        vis_detections = []
        
        for track in tracks:
            gid = track.get('global_id', -1)
            cls_id = track['class']
            class_name = Config.CLASS_MAP.get(cls_id, 'player')
            
            team_id = team_assignments.get(gid, -1) if cls_id in [Config.CLASS_ID_PLAYER, Config.CLASS_ID_GOALKEEPER] else -1
            
            vis_entry = {
                'global_id': gid,
                'track_id': track['track_id'],
                'bbox': track['bbox'],
                'class': class_name,
                'confidence': track['confidence'],
                'team_id': team_id,
                'tracked': True
            }
            
            vis_detections.append(vis_entry)
        
        # Draw frame
        annotated_frame = visualizer.draw_enhanced_frame(
            frame.copy(), vis_detections, team_assignments
        )
        
        # Add FPS
        current_fps = fps_counter.get_fps()
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                   (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
        
        writer.write(annotated_frame)
        
        # ========== CLEANUP & PROGRESS ==========
        cleanup_interval = getattr(Config, 'MEMORY_CLEANUP_INTERVAL', 100)
        if frame_idx % cleanup_interval == 0:
            id_manager.cleanup_inactive(frame_idx)
        
        if frame_idx % 10 == 0:
            progress.update(10)
            progress.print_progress()
        
        log_interval = getattr(Config, 'LOG_STATISTICS_INTERVAL', 100)
        if frame_idx % log_interval == 0:
            print()
            id_stats = id_manager.get_statistics()
            team_stats = team_classifier.get_team_stats()
            
            print(f"\n[Frame {frame_idx}/{total_frames}]")
            print(f"  FPS: {current_fps:.2f}")
            print(f"  Global IDs: {id_stats['active_ids']} active, {id_stats['total_ids_created']} total")
            print(f"  Team A: {team_stats['team_counts'].get(0, 0)} | Team B: {team_stats['team_counts'].get(1, 0)}")
    
    cap.release()
    writer.release()
    
    # Verify output was created
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        print(f"\n✓ Video saved: {output_path} ({file_size:.1f} MB)")
    else:
        print(f"\n✗ WARNING: Output video not found at {output_path}")
        print("  This might be a codec issue. Try:")
        print("  1. Check if 'output' folder exists")
        print("  2. Try output path without subdirectory: --out tracked.mp4")
    
    # ========== FINALIZE ==========
    
    print("\n\n" + "="*80)
    print("FINALIZING")
    print("="*80)
    
    analytics.calculate_final_metrics()
    
    export_mgr = ExportManager(output_dir=Config.EXPORT_DIRECTORY)
    
    if Config.EXPORT_DATA:
        print("Exporting data...")
        export_mgr.export_tracking_data(analytics, team_assignments)
        export_mgr.export_performance_metrics(analytics, team_assignments)
    
    if Config.GENERATE_HEATMAPS:
        print("Generating heatmaps...")
        analytics.generate_heatmaps(Config.EXPORT_DIRECTORY)
        export_mgr.generate_individual_player_heatmaps(analytics, team_assignments, (height, width))
    
    if Config.GENERATE_REPORTS:
        print("Generating reports...")
        analytics.generate_pass_network(Config.EXPORT_DIRECTORY)
        export_mgr.generate_performance_visualizations(analytics, team_assignments)
        export_mgr.generate_possession_analysis(analytics, team_assignments)
    
    id_manager.save_to_json(str(Path(Config.EXPORT_DIRECTORY) / "data" / "id_mappings.json"))
    export_mgr.save_all()
    
    # ========== SUMMARY ==========
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    id_stats = id_manager.get_statistics()
    team_stats = team_classifier.get_team_stats()
    
    print(f"\nGlobal IDs:")
    print(f"  Total created: {id_stats['total_ids_created']}")
    print(f"  Final active: {id_stats['active_ids']}")
    
    print(f"\nTeams:")
    print(f"  Team A: {team_stats['team_counts'].get(0, 0)} players")
    print(f"  Team B: {team_stats['team_counts'].get(1, 0)} players")
    
    print(f"\nPerformance:")
    print(f"  Frames: {frame_idx}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  FPS: {frame_idx/elapsed:.2f}")
    print(f"  Real-time factor: {(frame_idx/src_fps)/elapsed:.2f}x")
    
    print(f"\nOutputs:")
    print(f"  Video: {output_path}")
    print(f"  Analytics: {Config.EXPORT_DIRECTORY}/")
    
    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Football Tracking (BoT-SORT + SigLIP)")
    parser.add_argument("--video", "-v", type=str, required=True, help="Input video path")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output video path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    parser.add_argument("--tracker-config", type=str, default=None, help="BoT-SORT YAML config")
    parser.add_argument("--no-siglip", action="store_true", help="Disable SigLIP (use color-based)")
    
    args = parser.parse_args()
    
    try:
        run_enhanced_pipeline(
            video_path=args.video,
            output_path=args.out,
            max_frames=(args.max_frames or None),
            tracker_config=args.tracker_config,
            use_siglip=(not args.no_siglip)
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()