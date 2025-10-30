"""
Enhanced analytics with valuable insights for football tracking
Aligned to use team_id (0/1) and global_id consistently
"""
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from scipy.spatial.distance import euclidean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AdvancedAnalytics:
    """Analyzes player movements, passes, formations, possession, and generates statistics"""

    def __init__(self, field_dimensions=(105.0, 68.0), fps=25):
        self.field_length, self.field_width = field_dimensions
        self.fps = fps

        # Player tracking (keyed by global_id)
        self.player_positions = defaultdict(list)
        self.ball_positions = []
        self.pass_events = []
        self.sprint_events = []
        self.formation_snapshots = []

        # Heatmaps by team (0/1)
        self.team_heatmaps = {0: defaultdict(list), 1: defaultdict(list)}
        
        # Pass network by team (0/1)
        self.pass_network = {0: defaultdict(int), 1: defaultdict(int)}

        # Player metrics (keyed by global_id)
        self.player_metrics = defaultdict(lambda: {
            'total_distance': 0.0,
            'max_speed': 0.0,
            'avg_speed': 0.0,
            'sprints': 0,
            'passes_made': 0,
            'passes_received': 0,
            'pass_accuracy': 0.0,
            'successful_passes': 0,
            'failed_passes': 0,
            'time_in_possession': 0.0,
            'high_intensity_distance': 0.0,
            'touches': 0,
            'interceptions': 0,
            'tackles': 0
        })

        # NEW: Possession tracking
        self.possession_windows = []  # List of {frame, team_id, duration}
        self.current_possession = {'team_id': -1, 'start_frame': 0, 'player_id': -1}
        
        # NEW: Zone control (divide field into grid)
        self.zone_control = {0: defaultdict(int), 1: defaultdict(int)}  # team -> zone -> time
        
        # NEW: Transition events
        self.transitions = []  # {frame, from_team, to_team, type: 'turnover'/'interception'}
        
        # NEW: Pressing intensity
        self.pressing_events = []  # {frame, pressing_team, pressed_player}

        self.pixel_to_meter = None
        self.field_bounds = None
        
        self.current_frame_data = {
            'frame': 0,
            'timestamp': 0.0,
            'players': {},  # global_id -> record
            'ball': None
        }
        
        # Ball proximity tracking (for possession detection)
        self.ball_proximity_threshold = 50  # pixels

    def calibrate_field(self, frame_width, frame_height):
        """Calibrate pixel-to-meter conversion and field zones"""
        margin_x = int(frame_width * 0.08)
        margin_y = int(frame_height * 0.12)

        self.field_bounds = {
            'left': margin_x,
            'right': frame_width - margin_x,
            'top': margin_y,
            'bottom': frame_height - margin_y
        }

        field_width_pixels = self.field_bounds['right'] - self.field_bounds['left']
        self.pixel_to_meter = self.field_width / (field_width_pixels + 1e-6)

        # Define field zones (3x3 grid)
        self.zones = {}
        zone_width = (self.field_bounds['right'] - self.field_bounds['left']) / 3
        zone_height = (self.field_bounds['bottom'] - self.field_bounds['top']) / 3
        
        for i in range(3):
            for j in range(3):
                zone_id = i * 3 + j
                self.zones[zone_id] = {
                    'x_min': self.field_bounds['left'] + j * zone_width,
                    'x_max': self.field_bounds['left'] + (j + 1) * zone_width,
                    'y_min': self.field_bounds['top'] + i * zone_height,
                    'y_max': self.field_bounds['top'] + (i + 1) * zone_height
                }

        print(f"[Analytics] Calibrated: {self.pixel_to_meter:.5f} m/pixel, {len(self.zones)} zones")

    def update_positions(self, detections, team_assignments, frame_count, timestamp):
        """
        Update tracking with possession, zone control, and pressing detection
        
        Args:
            detections: dict {global_id -> {'bbox', 'class', ...}}
            team_assignments: dict {global_id -> team_id (0/1/-1)}
        """
        self.current_frame_data = {
            'frame': frame_count,
            'timestamp': timestamp,
            'players': {},
            'ball': None
        }

        ball_pos = None
        
        # Process all detections
        for gid, det in detections.items():
            try:
                global_id = int(gid)
            except:
                continue

            bbox = det.get('bbox')
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cls_id = det.get('class', 2)
            cls_name = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}.get(cls_id, 'player')

            if cls_name == 'ball':
                ball_pos = (cx, cy)
                ball_record = {
                    'x': cx, 'y': cy,
                    'frame': frame_count,
                    'timestamp': timestamp
                }
                self.ball_positions.append(ball_record)
                self.current_frame_data['ball'] = ball_record
            else:
                team_id = team_assignments.get(global_id, -1)
                
                player_record = {
                    'global_id': global_id,
                    'x': cx, 'y': cy,
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'team_id': team_id
                }
                
                # Calculate speed if we have previous position
                prev_positions = self.player_positions[global_id]
                if prev_positions:
                    prev = prev_positions[-1]
                    dist_px = euclidean((prev['x'], prev['y']), (cx, cy))
                    time_diff = timestamp - prev['timestamp']
                    if time_diff > 0:
                        dist_m = dist_px * (self.pixel_to_meter or 0.01)
                        speed_ms = dist_m / time_diff
                        speed_kmh = speed_ms * 3.6
                        player_record['speed'] = min(speed_kmh, 45)  # Cap unrealistic speeds
                
                self.player_positions[global_id].append(player_record)
                self.current_frame_data['players'][global_id] = player_record

                # Update heatmaps (only for valid teams)
                if team_id in [0, 1]:
                    self.team_heatmaps[team_id][global_id].append((cx, cy))
                    
                    # Update zone control
                    zone_id = self._get_zone(cx, cy)
                    if zone_id is not None:
                        self.zone_control[team_id][zone_id] += 1

        # Detect possession changes
        if ball_pos:
            self._update_possession(ball_pos, self.current_frame_data['players'], 
                                   frame_count, timestamp)
        
        # Detect pressing
        self._detect_pressing(self.current_frame_data['players'], ball_pos, frame_count)

        return self.current_frame_data

    def _get_zone(self, x, y):
        """Get zone ID for given position"""
        if not hasattr(self, 'zones'):
            return None
        for zone_id, bounds in self.zones.items():
            if (bounds['x_min'] <= x <= bounds['x_max'] and 
                bounds['y_min'] <= y <= bounds['y_max']):
                return zone_id
        return None

    def _update_possession(self, ball_pos, players, frame, timestamp):
        """Track which team has possession based on ball proximity"""
        closest_player = None
        min_dist = float('inf')
        
        for gid, p in players.items():
            if p['team_id'] not in [0, 1]:
                continue
            dist = euclidean(ball_pos, (p['x'], p['y']))
            if dist < min_dist:
                min_dist = dist
                closest_player = p
        
        if closest_player and min_dist < self.ball_proximity_threshold:
            new_team = closest_player['team_id']
            new_player = closest_player['global_id']
            
            # Track touches
            self.player_metrics[new_player]['touches'] += 1
            
            # Detect possession change (transition)
            if (self.current_possession['team_id'] != -1 and 
                self.current_possession['team_id'] != new_team):
                # Possession changed - record transition
                self.transitions.append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'from_team': self.current_possession['team_id'],
                    'to_team': new_team,
                    'from_player': self.current_possession['player_id'],
                    'to_player': new_player,
                    'type': 'turnover'
                })
                
                # Save possession window
                duration = frame - self.current_possession['start_frame']
                self.possession_windows.append({
                    'team_id': self.current_possession['team_id'],
                    'start_frame': self.current_possession['start_frame'],
                    'end_frame': frame,
                    'duration': duration
                })
            
            # Update current possession
            if self.current_possession['team_id'] != new_team:
                self.current_possession = {
                    'team_id': new_team,
                    'start_frame': frame,
                    'player_id': new_player
                }

    def _detect_pressing(self, players, ball_pos, frame):
        """Detect pressing situations (3+ opponents within radius of ball carrier)"""
        if not ball_pos:
            return
        
        # Find ball carrier
        carrier = None
        min_dist = float('inf')
        for gid, p in players.items():
            if p['team_id'] not in [0, 1]:
                continue
            dist = euclidean(ball_pos, (p['x'], p['y']))
            if dist < min_dist:
                min_dist = dist
                carrier = p
        
        if not carrier or min_dist > self.ball_proximity_threshold:
            return
        
        # Count opponents nearby
        press_radius = 100  # pixels
        opponents = [p for p in players.values() 
                    if p['team_id'] not in [-1, carrier['team_id']] and
                    euclidean((carrier['x'], carrier['y']), (p['x'], p['y'])) < press_radius]
        
        if len(opponents) >= 3:
            self.pressing_events.append({
                'frame': frame,
                'pressed_player': carrier['global_id'],
                'pressed_team': carrier['team_id'],
                'pressing_team': 1 - carrier['team_id'],  # opposite team
                'num_pressers': len(opponents)
            })

    def analyze_pass_event(self, passer_gid, receiver_gid, ball_pos, frame_data):
        """Analyze pass event with success/failure tracking"""
        passer_data = frame_data['players'].get(passer_gid)
        receiver_data = frame_data['players'].get(receiver_gid)

        if not passer_data or not receiver_data:
            return None

        pass_distance_px = euclidean(
            (passer_data['x'], passer_data['y']),
            (receiver_data['x'], receiver_data['y'])
        )
        pass_distance_m = pass_distance_px * (self.pixel_to_meter or 0.01)
        
        successful = (passer_data.get('team_id') == receiver_data.get('team_id'))

        pass_event = {
            'frame': frame_data['frame'],
            'timestamp': frame_data['timestamp'],
            'passer_id': passer_gid,
            'receiver_id': receiver_gid,
            'passer_team': passer_data.get('team_id', -1),
            'receiver_team': receiver_data.get('team_id', -1),
            'pass_distance': pass_distance_m,
            'ball_pos': ball_pos,
            'passer_pos': (passer_data['x'], passer_data['y']),
            'receiver_pos': (receiver_data['x'], receiver_data['y']),
            'successful': successful
        }

        self.pass_events.append(pass_event)

        # Update pass network
        if successful and passer_data.get('team_id') in [0, 1]:
            team = passer_data['team_id']
            self.pass_network[team][(passer_gid, receiver_gid)] += 1

        # Update metrics
        self.player_metrics[passer_gid]['passes_made'] += 1
        if successful:
            self.player_metrics[passer_gid]['successful_passes'] += 1
            self.player_metrics[receiver_gid]['passes_received'] += 1
        else:
            self.player_metrics[passer_gid]['failed_passes'] += 1

        return pass_event

    def calculate_final_metrics(self):
        """Calculate comprehensive final metrics"""
        print("[Analytics] Calculating final metrics...")

        # Calculate player movement metrics
        for gid, positions in self.player_positions.items():
            if len(positions) < 2:
                continue

            total_distance = 0.0
            high_intensity_dist = 0.0
            speeds = []

            for i in range(1, len(positions)):
                prev, curr = positions[i-1], positions[i]
                dist_px = euclidean((prev['x'], prev['y']), (curr['x'], curr['y']))
                dist_m = dist_px * (self.pixel_to_meter or 0.01)
                total_distance += dist_m

                time_diff = curr['timestamp'] - prev['timestamp']
                if time_diff > 0:
                    speed_ms = dist_m / time_diff
                    speed_kmh = speed_ms * 3.6
                    if speed_kmh < 45:
                        speeds.append(speed_kmh)
                        if speed_kmh > 15:  # High intensity threshold
                            high_intensity_dist += dist_m

            metrics = self.player_metrics[gid]
            metrics['total_distance'] = total_distance
            metrics['high_intensity_distance'] = high_intensity_dist
            metrics['max_speed'] = max(speeds) if speeds else 0
            metrics['avg_speed'] = float(np.mean(speeds)) if speeds else 0
            metrics['sprints'] = sum(1 for s in speeds if s > 20.0)

        # Calculate pass accuracy
        for gid in self.player_metrics.keys():
            made = self.player_metrics[gid]['passes_made']
            if made > 0:
                success = self.player_metrics[gid]['successful_passes']
                self.player_metrics[gid]['pass_accuracy'] = (success / made) * 100.0

        # Calculate possession percentages
        total_frames = sum(w['duration'] for w in self.possession_windows)
        if total_frames > 0:
            for team_id in [0, 1]:
                team_frames = sum(w['duration'] for w in self.possession_windows 
                                if w['team_id'] == team_id)
                pct = (team_frames / total_frames) * 100
                print(f"[Analytics] Team {team_id} possession: {pct:.1f}%")

    def generate_heatmaps(self, output_dir):
        """Generate team and zone control heatmaps"""
        out_dir = Path(output_dir) / "heatmaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        print("[Analytics] Generating heatmaps...")

        for team_id in [0, 1]:
            team_positions = []
            for gid, coords in self.team_heatmaps[team_id].items():
                team_positions.extend(coords)

            if not team_positions:
                continue

            arr = np.array(team_positions)
            heatmap, xedges, yedges = np.histogram2d(
                arr[:, 0], arr[:, 1], bins=50, density=True
            )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(heatmap.T, extent=extent, origin='lower',
                          cmap='hot', alpha=0.8)
            ax.set_title(f"Team {team_id} Position Heatmap")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            fig.colorbar(im, ax=ax, label='Density')
            fig.savefig(out_dir / f"team_{team_id}_heatmap.png", dpi=200, bbox_inches='tight')
            plt.close(fig)

    def generate_pass_network(self, output_dir):
        """Generate pass network visualizations"""
        out_dir = Path(output_dir) / "pass_networks"
        out_dir.mkdir(parents=True, exist_ok=True)
        print("[Analytics] Generating pass networks...")

        import networkx as nx

        for team_id in [0, 1]:
            if not self.pass_network[team_id]:
                continue

            G = nx.DiGraph()
            for (p, r), w in self.pass_network[team_id].items():
                G.add_edge(p, r, weight=w)

            if len(G.nodes) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, k=1.2, iterations=50)
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_w = max(weights) if weights else 1

            nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                   node_size=800, alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=[(w/max_w)*5 for w in weights],
                                   alpha=0.7, edge_color='gray', arrows=True)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

            ax.set_title(f"Team {team_id} Pass Network")
            ax.axis('off')
            fig.savefig(out_dir / f"team_{team_id}_pass_network.png",
                       dpi=200, bbox_inches='tight')
            plt.close(fig)