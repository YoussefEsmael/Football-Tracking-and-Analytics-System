"""
SigLIP + UMAP Team Classification
Uses vision-language model for robust jersey color understanding
"""
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import colorsys

try:
    import torch
    from transformers import AutoProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[SigLIP] transformers not available")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[SigLIP] umap-learn not available")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[SigLIP] scikit-learn not available")

from utils import safe_crop, normalize_features


class SigLIPTeamClassifier:
    """
    Team classification using SigLIP vision embeddings + UMAP dimensionality reduction
    Much more robust than color histograms for similar jerseys
    """
    
    def __init__(self, 
                 n_teams: int = 2,
                 warmup_frames: int = 100,
                 confidence_threshold: float = 0.2,
                 model_name: str = "google/siglip-base-patch16-224",
                 device: str = "cuda",
                 use_umap: bool = True,
                 umap_n_components: int = 8):
        """
        Initialize SigLIP + UMAP team classifier
        
        Args:
            n_teams: Number of teams (always 2)
            warmup_frames: Frames to collect before clustering
            confidence_threshold: Minimum confidence for assignment
            model_name: SigLIP model to use
            device: 'cuda' or 'cpu'
            use_umap: Enable UMAP dimensionality reduction
            umap_n_components: UMAP output dimensions
        """
        
        self.n_teams = n_teams
        self.warmup_frames = warmup_frames
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_umap = use_umap and UMAP_AVAILABLE
        
        # SigLIP model
        self.model = None
        self.processor = None
        self.siglip_available = False
        
        if TRANSFORMERS_AVAILABLE:
            self._load_siglip(model_name)
        
        # UMAP reducer
        self.umap_reducer = None
        self.umap_n_components = umap_n_components
        
        # Feature storage
        self.embeddings = defaultdict(list)  # global_id -> embeddings
        self.embedding_buffer_size = 20
        
        # Team assignments
        self.team_assignments = {}
        self.assignment_confidence = defaultdict(float)
        
        # Cluster models
        self.team_centroids = {}  # team_id -> centroid
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Team colors (for visualization)
        self.team_colors = {
            0: (0, 0, 255),      # Red
            1: (0, 255, 255),    # Yellow
            -1: (150, 150, 150)  # Unknown
        }
        
        self.is_initialized = False
        self.frame_count = 0
        
        print(f"[SigLIP] Team classifier initialized")
        print(f"  Model: {'SigLIP' if self.siglip_available else 'Fallback'}")
        print(f"  UMAP: {self.use_umap}")
        print(f"  Device: {self.device}")
    
    def _load_siglip(self, model_name: str):
        """Load SigLIP model"""
        try:
            print(f"[SigLIP] Loading model: {model_name}")
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.siglip_available = True
            print(f"[SigLIP] ✓ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"[SigLIP] ✗ Failed to load model: {e}")
            print("[SigLIP] Falling back to color-based features")
            self.siglip_available = False
    
    def extract_jersey_embedding(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """
        Extract SigLIP embedding from jersey region
        
        Args:
            frame: Input frame
            bbox: (x1, y1, x2, y2)
        
        Returns:
            Embedding vector or None
        """
        # Crop player region
        crop = safe_crop(frame, bbox, min_area=200)
        if crop is None:
            return None
        
        # Focus on torso (jersey region)
        h, w = crop.shape[:2]
        torso_top = int(h * 0.15)
        torso_bottom = int(h * 0.60)
        torso_left = int(w * 0.15)
        torso_right = int(w * 0.85)
        torso = crop[torso_top:torso_bottom, torso_left:torso_right]
        
        if torso.size == 0:
            torso = crop
        
        # Use SigLIP if available
        if self.siglip_available:
            return self._extract_siglip_embedding(torso)
        else:
            # Fallback to color features
            return self._extract_color_embedding(torso)
    
    def _extract_siglip_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract SigLIP vision embedding"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image_resized = cv2.resize(image_rgb, (224, 224))
            
            # Process image
            inputs = self.processor(images=image_resized, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy().flatten()
            
            # L2 normalize
            embedding = normalize_features(embedding)
            
            return embedding.astype(np.float32)
        
        except Exception as e:
            return None
    
    def _extract_color_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Fallback color-based embedding"""
        try:
            features = []
            
            # HSV histogram
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            
            features.extend(h_hist)
            features.extend(s_hist)
            
            # LAB dominant colors
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab_mean = np.mean(lab, axis=(0, 1))
            features.extend(lab_mean / 255.0)
            
            features = np.array(features, dtype=np.float32)
            return normalize_features(features)
        
        except Exception:
            return None
    
    def update(self, frame: np.ndarray, detections: Dict, frame_idx: int):
        """
        Update team classifications
        
        Args:
            frame: Current frame
            detections: Dict of global_id -> detection info
            frame_idx: Current frame index
        """
        self.frame_count = frame_idx
        
        # Extract embeddings for all players
        for gid, det in detections.items():
            if det.get('class', 2) != 2:  # Only players
                continue
            
            bbox = det.get('bbox')
            if bbox is None:
                continue
            
            # Extract embedding
            embedding = self.extract_jersey_embedding(frame, bbox)
            if embedding is not None:
                self.embeddings[gid].append(embedding)
                
                # Keep buffer size manageable
                if len(self.embeddings[gid]) > self.embedding_buffer_size:
                    self.embeddings[gid].pop(0)
        
        # Initialize teams after warmup
        enough_players = len([g for g, embs in self.embeddings.items() 
                             if len(embs) >= 3])
        
        if not self.is_initialized and enough_players >= self.n_teams * 5:
            self._initialize_teams()
        elif self.is_initialized:
            self._assign_new_players()
        
        # Periodic logging
        if frame_idx % 100 == 0 and self.is_initialized:
            stats = self.get_team_stats()
            print(f"[SigLIP] Frame {frame_idx}: "
                  f"Team A={stats['team_counts'].get(0, 0)}, "
                  f"Team B={stats['team_counts'].get(1, 0)}")
    
    def _initialize_teams(self):
        """Initialize team models using UMAP + clustering"""
        valid_gids = [gid for gid, embs in self.embeddings.items() if len(embs) >= 3]
        
        if len(valid_gids) < self.n_teams * 3:
            print(f"[SigLIP] Not enough players: {len(valid_gids)}")
            return
        
        # Aggregate embeddings
        X = []
        gids = []
        
        for gid in valid_gids:
            # Average embeddings for this player
            avg_emb = np.mean(self.embeddings[gid], axis=0)
            X.append(avg_emb)
            gids.append(gid)
        
        X = np.array(X)
        
        if X.shape[0] < self.n_teams:
            return
        
        # Apply UMAP dimensionality reduction
        if self.use_umap and UMAP_AVAILABLE:
            print(f"[SigLIP] Applying UMAP: {X.shape[1]} -> {self.umap_n_components} dims")
            
            self.umap_reducer = umap.UMAP(
                n_components=self.umap_n_components,
                n_neighbors=min(15, len(X) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            
            X_reduced = self.umap_reducer.fit_transform(X)
        else:
            X_reduced = X
        
        # Standardize
        if self.scaler and SKLEARN_AVAILABLE:
            X_scaled = self.scaler.fit_transform(X_reduced)
        else:
            X_scaled = X_reduced
        
        # Cluster into 2 teams using KMeans
        if SKLEARN_AVAILABLE:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
            labels = kmeans.fit_predict(X_scaled)
            
            # Assign teams
            for gid, label in zip(gids, labels):
                team_id = int(label) % 2
                self.team_assignments[gid] = team_id
                self.assignment_confidence[gid] = 0.8
            
            # Store centroids
            for team_id in [0, 1]:
                team_indices = [i for i, label in enumerate(labels) if label == team_id]
                if team_indices:
                    self.team_centroids[team_id] = np.mean(X_scaled[team_indices], axis=0)
            
            # Update visualization colors
            self._update_team_colors(gids, labels)
            
            self.is_initialized = True
            
            team_counts = [len([g for g in gids if self.team_assignments[g] == tid]) 
                          for tid in [0, 1]]
            
            print(f"[SigLIP] ✓ Initialized {len(valid_gids)} players into 2 teams:")
            print(f"  Team A: {team_counts[0]} players - Color: {self.team_colors[0]}")
            print(f"  Team B: {team_counts[1]} players - Color: {self.team_colors[1]}")
        
        else:
            print("[SigLIP] ✗ scikit-learn not available for clustering")
    
    def _assign_new_players(self):
        """Assign teams to new players"""
        if not self.team_centroids:
            return
        
        for gid, embs in self.embeddings.items():
            # Skip already confident assignments
            if gid in self.team_assignments and self.assignment_confidence[gid] >= 0.8:
                continue
            
            if len(embs) < 3:
                if gid not in self.team_assignments:
                    self.team_assignments[gid] = -1
                continue
            
            # Get average embedding
            avg_emb = np.mean(embs, axis=0)
            
            # Transform with UMAP if available
            if self.umap_reducer:
                avg_emb_transformed = self.umap_reducer.transform([avg_emb])[0]
            else:
                avg_emb_transformed = avg_emb
            
            # Standardize
            if self.scaler:
                avg_emb_scaled = self.scaler.transform([avg_emb_transformed])[0]
            else:
                avg_emb_scaled = avg_emb_transformed
            
            # Find closest team centroid
            best_team = -1
            best_dist = float('inf')
            
            for team_id, centroid in self.team_centroids.items():
                dist = np.linalg.norm(avg_emb_scaled - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_team = team_id
            
            # Assign if distance is reasonable
            # Convert distance to similarity (closer = more similar)
            similarity = 1.0 / (1.0 + best_dist)
            
            if similarity > 0.25:  # Threshold
                self.team_assignments[gid] = best_team
                prev_conf = self.assignment_confidence[gid]
                self.assignment_confidence[gid] = 0.7 * prev_conf + 0.3 * similarity
            else:
                self.team_assignments[gid] = -1
    
    def _update_team_colors(self, gids: List[int], labels: np.ndarray):
        """Update team display colors based on dominant jersey colors"""
        for team_id in [0, 1]:
            team_gids = [gids[i] for i, label in enumerate(labels) if label == team_id]
            
            if not team_gids:
                continue
            
            # Get average hue from embeddings (approximate)
            # This is a heuristic - SigLIP doesn't directly give color
            hues = []
            
            for gid in team_gids[:5]:  # Sample first 5 players
                if self.embeddings[gid]:
                    # Use color fallback to get hue
                    # In production, you'd extract this during embedding
                    pass
            
            # For now, use distinct colors
            if team_id == 0:
                self.team_colors[0] = (0, 0, 255)  # Red
            else:
                self.team_colors[1] = (255, 255, 0)  # Cyan
    
    def get_team_assignment(self, global_id: int) -> int:
        """Get team assignment for global ID"""
        team_id = self.team_assignments.get(global_id, -1)
        if team_id not in [0, 1, -1]:
            return -1
        return team_id
    
    def get_team_color(self, team_id: int) -> Tuple[int, int, int]:
        """Get team color (BGR)"""
        return self.team_colors.get(team_id, (200, 200, 200))
    
    def get_team_stats(self) -> Dict:
        """Get team assignment statistics"""
        stats = {
            'total_tracked': len([g for g, t in self.team_assignments.items() if t != -1]),
            'team_counts': {},
            'avg_confidence': {}
        }
        
        for team_id in [0, 1]:
            gids = [g for g, t in self.team_assignments.items() if t == team_id]
            stats['team_counts'][team_id] = len(gids)
            if gids:
                stats['avg_confidence'][team_id] = float(
                    np.mean([self.assignment_confidence[g] for g in gids])
                )
        
        return stats


# Installation helper
def print_installation_instructions():
    """Print installation instructions for dependencies"""
    print("\n" + "="*60)
    print("SigLIP + UMAP Team Classifier Dependencies")
    print("="*60)
    print("\nRequired packages:")
    print("  pip install transformers")
    print("  pip install umap-learn")
    print("  pip install scikit-learn")
    print("\nFor GPU support:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("="*60 + "\n")


if __name__ == "__main__":
    if not all([TRANSFORMERS_AVAILABLE, UMAP_AVAILABLE, SKLEARN_AVAILABLE]):
        print_installation_instructions()
    else:
        print("✓ All dependencies available")
        
        # Test initialization
        classifier = SigLIPTeamClassifier(
            model_name="google/siglip-base-patch16-224",
            use_umap=True,
            device='cuda'
        )
        print("✓ SigLIP team classifier initialized")