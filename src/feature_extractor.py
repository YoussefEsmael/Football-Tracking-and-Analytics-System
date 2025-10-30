"""
Unified feature extraction for both color and deep ReID features
Used by team classifier and ID manager
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from collections import deque
import time

from utils import safe_crop, normalize_features

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    T = None
    TORCH_AVAILABLE = False


class FeatureExtractor:
    """
    Unified feature extraction with color and deep features
    Handles batching for efficiency
    """
    
    def __init__(self, deep_model_path: Optional[str] = None, 
                 device: str = 'cuda',
                 enable_deep: bool = True):
        self.device = device if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.enable_deep = enable_deep and TORCH_AVAILABLE
        
        self.deep_model = None
        self.transform = None
        
        # Performance tracking
        self.color_extraction_times = deque(maxlen=100)
        self.deep_extraction_times = deque(maxlen=100)
        
        if self.enable_deep and deep_model_path:
            self._load_deep_model(deep_model_path)
        
        if not self.enable_deep:
            print("[Features] Using color features only")
    
    def _load_deep_model(self, model_path: str):
        """Load deep ReID model"""
        try:
            from torchvision.models import resnet50
            
            print(f"[Features] Loading deep model from {model_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Create model
            self.deep_model = resnet50(pretrained=False)
            self.deep_model.fc = nn.Identity()  # Remove classification layer
            
            # Load weights flexibly
            try:
                self.deep_model.load_state_dict(state_dict, strict=False)
            except:
                # Try removing 'module.' prefix
                new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.deep_model.load_state_dict(new_state, strict=False)
            
            self.deep_model.to(self.device)
            self.deep_model.eval()
            
            # Setup transform
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),  # Standard person ReID size
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print(f"[Features] ✓ Deep model loaded on {self.device}")
            
        except Exception as e:
            print(f"[Features] ✗ Failed to load deep model: {e}")
            self.deep_model = None
            self.enable_deep = False
    
    def extract_color_features(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """
        Extract color-based features from jersey region
        
        Returns:
            32-dimensional feature vector or None
        """
        start_time = time.time()
        
        try:
            # Crop image
            crop = safe_crop(frame, bbox, min_area=100)
            if crop is None:
                return None
            
            h, w = crop.shape[:2]
            
            # Focus on torso region (jersey)
            torso_top = int(h * 0.15)
            torso_bottom = int(h * 0.55)
            torso_left = int(w * 0.2)
            torso_right = int(w * 0.8)
            torso = crop[torso_top:torso_bottom, torso_left:torso_right]
            
            if torso.size == 0:
                torso = crop
            
            features = []
            
            # HSV color histograms
            try:
                hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
                h_channel = hsv[:, :, 0]
                s_channel = hsv[:, :, 1]
                v_channel = hsv[:, :, 2]
                
                # Filter out non-jersey colors (grass, skin, etc.)
                mask = ((h_channel < 40) | (h_channel > 80)) & \
                       (s_channel > 40) & \
                       (v_channel > 40) & (v_channel < 230)
                
                if np.sum(mask) > 50:
                    # Hue histogram (18 bins)
                    hist_h = cv2.calcHist([hsv], [0], mask.astype(np.uint8), [18], [0, 180])
                    hist_h = cv2.normalize(hist_h, hist_h).flatten()
                    features.extend(hist_h)
                    
                    # Saturation histogram (8 bins)
                    hist_s = cv2.calcHist([hsv], [1], mask.astype(np.uint8), [8], [0, 256])
                    hist_s = cv2.normalize(hist_s, hist_s).flatten()
                    features.extend(hist_s)
                else:
                    features.extend(np.zeros(26))
            except:
                features.extend(np.zeros(26))
            
            # LAB color clustering
            try:
                lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
                pixels = lab.reshape(-1, 3)
                
                if len(pixels) > 20:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=3)
                    kmeans.fit(pixels)
                    
                    # Get top 2 cluster centers
                    labels, counts = np.unique(kmeans.labels_, return_counts=True)
                    sorted_clusters = sorted(
                        zip(kmeans.cluster_centers_, counts),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    for center, _ in sorted_clusters[:2]:
                        features.extend(center / 255.0)
                else:
                    features.extend(np.zeros(6))
            except:
                features.extend(np.zeros(6))
            
            # Ensure consistent dimension
            features = np.array(features, dtype=np.float32)
            if len(features) < 32:
                features = np.pad(features, (0, 32 - len(features)), 'constant')
            else:
                features = features[:32]
            
            self.color_extraction_times.append(time.time() - start_time)
            
            return normalize_features(features)
        
        except Exception as e:
            return None
    
    def extract_deep_features(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """
        Extract deep ReID features
        
        Returns:
            2048-dimensional feature vector or None
        """
        if not self.enable_deep or self.deep_model is None:
            return None
        
        start_time = time.time()
        
        try:
            # Crop image
            crop = safe_crop(frame, bbox, min_area=100)
            if crop is None:
                return None
            
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Transform and add batch dimension
            input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.deep_model(input_tensor)
                
                # Handle different output formats
                if isinstance(features, (tuple, list)):
                    features = features[0]
                
                # Flatten if needed
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                
                # L2 normalize
                features = F.normalize(features, p=2, dim=1)
                features = features.cpu().numpy().flatten()
            
            self.deep_extraction_times.append(time.time() - start_time)
            
            return features.astype(np.float32)
        
        except Exception as e:
            return None
    
    def extract_features_batch(self, frame: np.ndarray, 
                              bboxes: List[Tuple]) -> Tuple[List, List]:
        """
        Extract features for multiple bboxes in batch (more efficient)
        
        Returns:
            (color_features, deep_features) lists
        """
        color_features = []
        deep_features = []
        
        if not self.enable_deep or self.deep_model is None:
            # No batching for color only
            for bbox in bboxes:
                color_feat = self.extract_color_features(frame, bbox)
                color_features.append(color_feat)
                deep_features.append(None)
            return color_features, deep_features
        
        # Extract color features (not batchable)
        for bbox in bboxes:
            color_feat = self.extract_color_features(frame, bbox)
            color_features.append(color_feat)
        
        # Batch extract deep features
        crops = []
        valid_indices = []
        
        for i, bbox in enumerate(bboxes):
            crop = safe_crop(frame, bbox, min_area=100)
            if crop is not None:
                crops.append(crop)
                valid_indices.append(i)
        
        if not crops:
            return color_features, [None] * len(bboxes)
        
        try:
            # Prepare batch
            batch_tensors = []
            for crop in crops:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self.transform(crop_rgb)
                batch_tensors.append(tensor)
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Extract features in batch
            with torch.no_grad():
                batch_features = self.deep_model(batch)
                
                if isinstance(batch_features, (tuple, list)):
                    batch_features = batch_features[0]
                
                if len(batch_features.shape) > 2:
                    batch_features = batch_features.view(batch_features.size(0), -1)
                
                batch_features = F.normalize(batch_features, p=2, dim=1)
                batch_features = batch_features.cpu().numpy()
            
            # Map back to original indices
            deep_features = [None] * len(bboxes)
            for i, idx in enumerate(valid_indices):
                deep_features[idx] = batch_features[i].astype(np.float32)
        
        except Exception as e:
            print(f"[Features] Batch extraction error: {e}")
            deep_features = [None] * len(bboxes)
        
        return color_features, deep_features
    
    def get_statistics(self) -> dict:
        """Get performance statistics"""
        stats = {
            'enable_deep': self.enable_deep,
            'device': self.device
        }
        
        if self.color_extraction_times:
            stats['avg_color_time_ms'] = np.mean(self.color_extraction_times) * 1000
        
        if self.deep_extraction_times:
            stats['avg_deep_time_ms'] = np.mean(self.deep_extraction_times) * 1000
        
        return stats