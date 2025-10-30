"""
Simplified Global ID Management
BoT-SORT handles ReID internally, this just maintains global ID mapping
and handles track ID persistence across video
"""
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime


class SimpleGlobalIDManager:
    """
    Lightweight ID manager that works with BoT-SORT track IDs
    No feature extraction - BoT-SORT handles all ReID internally
    Just maintains persistent global IDs and metadata
    """
    
    def __init__(self, spatial_threshold=200, temporal_window=90):
        # ID mapping: BoT-SORT track_id -> global_id
        self._track_to_global = {}
        self._next_global_id = 1
        
        # Track metadata
        self._global_metadata = {}  # global_id -> metadata
        self._last_seen = {}  # global_id -> frame_idx
        self._last_positions = {}  # global_id -> (x, y)
        
        # Configuration
        self.spatial_threshold = spatial_threshold
        self.temporal_window = temporal_window
        
        # Statistics
        self.stats = {
            'total_ids_created': 0,
            'track_assignments': 0,
            'tracks_mapped': 0
        }
    
    def get_or_create_global_id(self, track_id: int, bbox: Tuple,
                                frame_idx: int, class_id: int = 2) -> Tuple[int, float]:
        """
        Map BoT-SORT track_id to persistent global_id
        
        Args:
            track_id: BoT-SORT track ID
            bbox: (x1, y1, x2, y2)
            frame_idx: Current frame index
            class_id: Object class
        
        Returns:
            (global_id, confidence=1.0)
        """
        
        # Check if track already has global ID
        if track_id in self._track_to_global:
            global_id = self._track_to_global[track_id]
            self._update_metadata(global_id, bbox, frame_idx)
            return global_id, 1.0
        
        # Create new global ID for new track
        global_id = self._next_global_id
        self._next_global_id += 1
        
        self._track_to_global[track_id] = global_id
        
        # Initialize metadata
        center = self._get_bbox_center(bbox)
        self._global_metadata[global_id] = {
            'first_seen': frame_idx,
            'last_seen': frame_idx,
            'total_frames': 1,
            'class': class_id,
            'botsort_track_id': track_id
        }
        self._last_seen[global_id] = frame_idx
        self._last_positions[global_id] = center
        
        self.stats['total_ids_created'] += 1
        self.stats['track_assignments'] += 1
        self.stats['tracks_mapped'] += 1
        
        return global_id, 1.0
    
    def _get_bbox_center(self, bbox: Tuple) -> Tuple[float, float]:
        """Calculate bbox center"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _update_metadata(self, global_id: int, bbox: Tuple, frame_idx: int):
        """Update metadata for global ID"""
        center = self._get_bbox_center(bbox)
        
        if global_id in self._global_metadata:
            self._global_metadata[global_id]['last_seen'] = frame_idx
            self._global_metadata[global_id]['total_frames'] += 1
        
        self._last_seen[global_id] = frame_idx
        self._last_positions[global_id] = center
    
    def cleanup_inactive(self, current_frame: int):
        """Remove inactive global IDs to save memory"""
        inactive = []
        
        for global_id, last_frame in list(self._last_seen.items()):
            if current_frame - last_frame > self.temporal_window * 2:
                inactive.append(global_id)
        
        for global_id in inactive:
            # Find and remove track mapping
            track_ids_to_remove = [tid for tid, gid in self._track_to_global.items() 
                                  if gid == global_id]
            for tid in track_ids_to_remove:
                del self._track_to_global[tid]
            
            # Remove from tracking dicts but keep metadata for export
            self._last_seen.pop(global_id, None)
            self._last_positions.pop(global_id, None)
    
    def get_global_id(self, track_id: int) -> Optional[int]:
        """Get global ID for track ID"""
        return self._track_to_global.get(track_id)
    
    def get_metadata(self, global_id: int) -> Dict:
        """Get metadata for global ID"""
        return self._global_metadata.get(global_id, {})
    
    def get_all_active_ids(self) -> List[int]:
        """Get all currently active global IDs"""
        return list(self._last_seen.keys())
    
    def get_statistics(self) -> Dict:
        """Get ID management statistics"""
        return {
            **self.stats,
            'active_ids': len(self._last_seen),
            'total_tracks': len(self._track_to_global)
        }
    
    def save_to_json(self, filepath: str):
        """Save ID mappings and metadata to JSON"""
        data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_global_ids': self._next_global_id - 1,
                'statistics': self.get_statistics(),
                'note': 'BoT-SORT handles ReID internally'
            },
            'id_mappings': {
                f"track_{k}": v for k, v in self._track_to_global.items()
            },
            'player_metadata': {
                str(k): {
                    'first_seen': v['first_seen'],
                    'last_seen': v['last_seen'],
                    'total_frames': v['total_frames'],
                    'class': v['class'],
                    'botsort_track_id': v.get('botsort_track_id', -1)
                }
                for k, v in self._global_metadata.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[IDManager] Saved to {filepath}")