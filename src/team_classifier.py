"""
Fixed Advanced team classification (global_id-based) - ONLY 2 TEAMS
"""
import cv2
import numpy as np
import colorsys
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class AdvancedTeamClassifier:
    """Classifies players into teams based on jersey colors + optional ReID features"""

    def __init__(self, n_teams=2, warmup_frames=30, confidence_threshold=0.7, use_reid_features=True):
        self.n_teams = n_teams
        self.warmup_frames = warmup_frames
        self.confidence_threshold = confidence_threshold
        self.use_reid_features = use_reid_features

        # store per global_id
        self.color_features = defaultdict(list)
        self.reid_features = defaultdict(list)
        self.spatial_features = defaultdict(list)

        self.team_assignments = {}          # global_id → team_id (0 or 1)
        self.assignment_confidence = defaultdict(float)

        # learned models
        self.team_color_models = {}
        self.team_reid_models = {}

        # FIXED: Only 2 team colors + unknown + goalkeeper
        self.team_colors = {
            0: (255, 50, 50),      # Team A - Red
            1: (50, 50, 255),      # Team B - Blue
            -1: (200, 200, 200)    # Unknown - Gray
        }

        self.frame_count = 0
        self.is_initialized = False

    def extract_jersey_features(self, frame, bbox):
        """Extract color features from player's jersey region"""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return None

        h, w = player_crop.shape[:2]
        torso_top = int(h * 0.15)
        torso_bottom = int(h * 0.55)
        torso_left = int(w * 0.2)
        torso_right = int(w * 0.8)
        torso = player_crop[torso_top:torso_bottom, torso_left:torso_right]

        if torso.size == 0:
            torso = player_crop

        features = []

        # HSV histograms
        try:
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            mask = ((h_channel < 40) | (h_channel > 80)) & (s_channel > 40) & (v_channel > 40) & (v_channel < 230)

            if np.sum(mask) > 50:
                hist_h = cv2.calcHist([hsv], [0], mask.astype(np.uint8), [18], [0, 180])
                hist_h = cv2.normalize(hist_h, hist_h).flatten()
                features.extend(hist_h)

                hist_s = cv2.calcHist([hsv], [1], mask.astype(np.uint8), [8], [0, 256])
                hist_s = cv2.normalize(hist_s, hist_s).flatten()
                features.extend(hist_s)
            else:
                features.extend(np.zeros(26))
        except Exception:
            features.extend(np.zeros(26))

        # LAB clustering
        try:
            lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
            pixels = lab.reshape(-1, 3)

            if len(pixels) > 20:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=3)
                kmeans.fit(pixels)
                labels, counts = np.unique(kmeans.labels_, return_counts=True)
                sorted_clusters = sorted(zip(kmeans.cluster_centers_, counts), key=lambda x: x[1], reverse=True)

                for center, _ in sorted_clusters[:2]:
                    features.extend(center / 255.0)
            else:
                features.extend(np.zeros(6))
        except Exception:
            features.extend(np.zeros(6))

        return np.array(features, dtype=np.float32)

    def update(self, frame, detections, frame_idx, reid_features_dict=None):
        """
        Update team classifications with new frame data.
        `detections` should be a dict: global_id → det
        Each det should include { 'track_id', 'bbox', 'class', 'global_id' }
        """
        self.frame_count = frame_idx

        for gid, det in detections.items():
            if det.get('class', 2) != 2:  # Only players (not goalkeepers)
                continue

            bbox = det.get('bbox')
            if bbox is None:
                continue

            # Color features
            features = self.extract_jersey_features(frame, bbox)
            if features is not None:
                self.color_features[gid].append(features)
                if len(self.color_features[gid]) > 15:
                    self.color_features[gid].pop(0)

            # ReID features
            if reid_features_dict and gid in reid_features_dict:
                reid_feat = reid_features_dict[gid]
                if reid_feat is not None and len(reid_feat) > 0:
                    self.reid_features[gid].append(reid_feat)
                    if len(self.reid_features[gid]) > 10:
                        self.reid_features[gid].pop(0)

            # Spatial features
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            self.spatial_features[gid].append((cx, cy))
            if len(self.spatial_features[gid]) > 10:
                self.spatial_features[gid].pop(0)

        # Dynamic warmup - need enough players before clustering
        enough_players = len([g for g, feats in self.color_features.items() if len(feats) >= 3])
        if not self.is_initialized and enough_players >= self.n_teams * 5:
            self._initialize_teams()
        elif self.is_initialized:
            self._assign_new_tracks()

        # Periodic logging
        if frame_idx % 100 == 0 and self.is_initialized:
            stats = self.get_team_stats()
            print(f"[TeamClassifier] Frame {frame_idx}: Team A={stats['team_counts'].get(0, 0)}, Team B={stats['team_counts'].get(1, 0)}")

    def _initialize_teams(self):
        """Initialize team models using collected features - FORCE 2 TEAMS ONLY"""
        valid_gids = [gid for gid, feats in self.color_features.items() if len(feats) >= 3]

        if len(valid_gids) < self.n_teams * 3:
            print(f"[TeamClassifier] Not enough players ({len(valid_gids)}) for initialization")
            return

        track_features, gids = [], []
        for gid in valid_gids:
            avg_color = np.mean(self.color_features[gid], axis=0)

            if self.use_reid_features and gid in self.reid_features and len(self.reid_features[gid]) > 0:
                avg_reid = np.mean(self.reid_features[gid], axis=0)
                avg_reid = avg_reid / (np.linalg.norm(avg_reid) + 1e-6)
                combined = np.concatenate([avg_color * 0.6, avg_reid * 0.4])
            else:
                combined = avg_color

            track_features.append(combined)
            gids.append(gid)

        X = np.array(track_features)
        if X.shape[0] < self.n_teams:
            print("[TeamClassifier] Not enough samples for clustering")
            return

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # FORCE exactly 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
        except Exception as e:
            print(f"[TeamClassifier] KMeans failed: {e}")
            return

        # Assign each player to team 0 or 1
        for gid, label in zip(gids, labels):
            team_id = int(label) % 2  # Force 0 or 1
            self.team_assignments[gid] = team_id
            self.assignment_confidence[gid] = 1.0

        # Build team color models for ONLY 2 teams
        for team_id in [0, 1]:
            team_gids = [g for g, t in self.team_assignments.items() if t == team_id]
            if not team_gids:
                continue

            team_color_feats = [f for g in team_gids for f in self.color_features[g]]
            if team_color_feats:
                self.team_color_models[team_id] = np.mean(team_color_feats, axis=0)
                # Update team display color based on dominant jersey color
                dominant_color = self._get_dominant_color_from_features(self.team_color_models[team_id])
                self.team_colors[team_id] = dominant_color

            if self.use_reid_features:
                team_reid_feats = [f for g in team_gids for f in self.reid_features.get(g, [])]
                if team_reid_feats:
                    self.team_reid_models[team_id] = np.mean(team_reid_feats, axis=0)

        self.is_initialized = True
        print(f"[TeamClassifier] Initialized {len(valid_gids)} players into 2 teams:")
        print(f"  Team A: {len([g for g in gids if self.team_assignments[g] == 0])} players - Color: {self.team_colors[0]}")
        print(f"  Team B: {len([g for g in gids if self.team_assignments[g] == 1])} players - Color: {self.team_colors[1]}")

    def _get_dominant_color_from_features(self, features):
        """Extract dominant BGR color from feature vector"""
        hue_hist = features[:18]
        if len(hue_hist) == 0 or np.sum(hue_hist) == 0:
            return (200, 200, 200)
        dominant_hue_bin = np.argmax(hue_hist)
        dominant_hue = (dominant_hue_bin * 10) / 180.0
        rgb = colorsys.hsv_to_rgb(dominant_hue, 0.8, 0.8)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

    def _assign_new_tracks(self):
        """Assign teams to new or unassigned global_ids - ONLY to team 0 or 1"""
        if not self.team_color_models:
            return

        for gid, feats in self.color_features.items():
            # Persistence lock - don't reassign confident assignments
            if gid in self.team_assignments and self.assignment_confidence[gid] >= 0.8:
                continue

            if len(feats) < 3:
                if gid not in self.team_assignments:
                    self.team_assignments[gid] = -1
                continue

            avg_color = np.mean(feats, axis=0)
            best_team, best_similarity = -1, -1

            # Only check against team 0 and 1
            for team_id in [0, 1]:
                if team_id not in self.team_color_models:
                    continue

                color_model = self.team_color_models[team_id]
                denom = (np.linalg.norm(avg_color) * np.linalg.norm(color_model) + 1e-8)
                color_sim = float(np.dot(avg_color, color_model) / denom)

                reid_sim = 0.0
                if self.use_reid_features and gid in self.reid_features and team_id in self.team_reid_models:
                    avg_reid = np.mean(self.reid_features[gid], axis=0)
                    denom = (np.linalg.norm(avg_reid) * np.linalg.norm(self.team_reid_models[team_id]) + 1e-8)
                    reid_sim = float(np.dot(avg_reid, self.team_reid_models[team_id]) / denom)

                similarity = 0.6 * color_sim + 0.4 * reid_sim if reid_sim > 0 else color_sim
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_team = team_id

            if best_similarity > 0.55:
                self.team_assignments[gid] = best_team
                prev_conf = self.assignment_confidence[gid]
                self.assignment_confidence[gid] = 0.7 * prev_conf + 0.3 * best_similarity
            else:
                if gid in self.team_assignments and self.team_assignments[gid] != -1:
                    self.assignment_confidence[gid] *= 0.95
                else:
                    self.team_assignments[gid] = -1

    def get_team_assignment(self, global_id):
        """Returns 0, 1, or -1"""
        team_id = self.team_assignments.get(global_id, -1)
        # Safety: force to valid range
        if team_id not in [0, 1, -1]:
            return -1
        return team_id

    def get_team_color(self, team_id):
        """Returns BGR color tuple for team 0, 1, or -1 (unknown)"""
        return self.team_colors.get(team_id, (200, 200, 200))

    def get_team_stats(self):
        """Returns statistics about team assignments"""
        stats = {
            "total_tracked": len([g for g, t in self.team_assignments.items() if t != -1]),
            "team_counts": {},
            "avg_confidence": {}
        }
        
        # Only report teams 0 and 1
        for team_id in [0, 1]:
            gids = [g for g, t in self.team_assignments.items() if t == team_id]
            stats["team_counts"][team_id] = len(gids)
            if gids:
                stats["avg_confidence"][team_id] = float(np.mean([self.assignment_confidence[g] for g in gids]))
        
        return stats

    def predict_team_from_features(self, features):
        """
        Given jersey color features, return most likely team_id (0 or 1) or -1
        """
        if features is None or not self.team_color_models:
            return -1

        avg_feat = np.array(features, dtype=np.float32)
        best_team = -1
        best_sim = -1.0
        
        # Only check against teams 0 and 1
        for team_id in [0, 1]:
            if team_id not in self.team_color_models:
                continue
                
            model_arr = np.array(self.team_color_models[team_id], dtype=np.float32)
            min_len = min(len(model_arr), len(avg_feat))
            if min_len == 0:
                continue
                
            sim = float(np.dot(avg_feat[:min_len], model_arr[:min_len]) /
                       (np.linalg.norm(avg_feat[:min_len]) * np.linalg.norm(model_arr[:min_len]) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_team = team_id

        return best_team if best_sim >= 0.45 else -1