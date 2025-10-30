"""
visualizer.py - FIXED VERSION
Enhanced EllipticalVisualizer with safety checks
"""
import cv2
import numpy as np
from collections import defaultdict, deque


class EllipticalVisualizer:
    def __init__(self, team_classifier=None):
        self.team_classifier = team_classifier
        self.ellipse_thickness = -1
        self.ellipse_alpha = 0.25
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 0.5
        self.text_thickness = 2

        # Trails for movement history
        self.track_trails = defaultdict(lambda: deque(maxlen=25))

        # Special colors
        self.goalkeeper_color = (0, 215, 255)  # gold
        self.referee_color = (255, 255, 0)     # cyan-yellow

    # ---------------------------
    # Utility: Text with shadow
    # ---------------------------
    def draw_text_with_shadow(self, frame, text, pos, font_scale=0.5,
                              color=(255, 255, 255), thickness=1):
        x, y = pos
        cv2.putText(frame, text, (x+1, y+1),
                    self.text_font, font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    self.text_font, font_scale, color, thickness, cv2.LINE_AA)
        return frame

    # ---------------------------
    # Player ellipse & glow - WITH SAFETY CHECKS
    # ---------------------------
    def draw_player_ellipse(self, frame, bbox, global_id, team_id,
                            confidence=1.0, is_goalkeeper=False):
        """Draw glowing ellipse + badge for player - WITH SAFETY CHECKS"""
        
        # Validate bbox
        if bbox is None or len(bbox) != 4:
            return frame
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure valid dimensions
        if x2 <= x1 or y2 <= y1:
            return frame
        
        width = x2 - x1
        height = y2 - y1
        
        # Skip if too small
        if width < 5 or height < 5:
            return frame
        
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Calculate ellipse axes with safety
        axes_w = int(width * 0.55)
        axes_h = int(height * 0.48)
        
        # Ensure positive axes
        if axes_w <= 0 or axes_h <= 0:
            return frame
        
        axes = (axes_w, axes_h)

        # -------------------------------
        # Pick base color
        # -------------------------------
        if is_goalkeeper:
            base_color = self.goalkeeper_color
        else:
            if team_id != -1:
                base_color = self.team_classifier.get_team_color(team_id)
            else:
                base_color = (200, 200, 200)

        # Heartbeat glow
        pulse = int(np.sin(cv2.getTickCount() * 0.00025) * 8 + 18)
        
        # Draw glow layers
        for i in range(3):
            overlay = frame.copy()
            grow_w = axes[0] + i*4 + pulse//3
            grow_h = axes[1] + i*2 + pulse//4
            
            # Safety check for grown axes
            if grow_w > 0 and grow_h > 0:
                grow_axes = (grow_w, grow_h)
                alpha = max(0.12 - i*0.03, 0.03)
                try:
                    cv2.ellipse(overlay, (cx, cy), grow_axes, 0, 0, 360, base_color, -1)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                except cv2.error:
                    pass  # Skip if still invalid

        # Solid ellipse core
        overlay = frame.copy()
        try:
            cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, base_color, -1)
            frame = cv2.addWeighted(overlay, self.ellipse_alpha, frame, 1 - self.ellipse_alpha, 0)
        except cv2.error:
            # Fallback to simple circle if ellipse fails
            radius = min(axes[0], axes[1])
            if radius > 0:
                cv2.circle(overlay, (cx, cy), radius, base_color, -1)
                frame = cv2.addWeighted(overlay, self.ellipse_alpha, frame, 1 - self.ellipse_alpha, 0)

        # Badge above head
        badge_y = y1 - 18
        
        # Ensure badge is on screen
        if badge_y < 10:
            badge_y = y1 + 10
        
        number = str(global_id) if global_id != -1 else "?"
        badge_center = (cx, badge_y)
        badge_radius = 14 if is_goalkeeper else 12
        
        try:
            cv2.circle(frame, badge_center, badge_radius, base_color, -1)
            cv2.circle(frame, badge_center, badge_radius, (255, 255, 255), 2)
            
            text_size = cv2.getTextSize(number, self.text_font, 0.5, 1)[0]
            text_x = badge_center[0] - text_size[0] // 2
            text_y = badge_center[1] + text_size[1] // 2

            self.draw_text_with_shadow(frame, number, (text_x, text_y),
                                       font_scale=0.5, color=(255, 255, 255), thickness=1)

            if is_goalkeeper:
                self.draw_text_with_shadow(frame, "GOALKEEPER",
                                           (cx - 35, badge_y - 20),
                                           font_scale=0.45, color=base_color, thickness=1)
        except Exception:
            pass  # Skip badge if any drawing fails

        # Update trail
        self.track_trails[global_id].append((cx, cy))
        
        return frame

    # ---------------------------
    # Trails
    # ---------------------------
    def draw_track_trail(self, frame, global_id, team_id, is_goalkeeper=False):
        trail = self.track_trails.get(global_id, [])
        if len(trail) < 2:
            return frame

        if is_goalkeeper:
            color = self.goalkeeper_color
        else:
            if team_id != -1:
                color = self.team_classifier.get_team_color(team_id)
            else:
                color = (180, 180, 180)

        for i in range(1, len(trail)):
            alpha = i / len(trail)
            thickness = int(1 + alpha * 2)
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, trail[i - 1], trail[i], faded_color, thickness, cv2.LINE_AA)
        return frame

    # ---------------------------
    # Ball highlight (glow pulse)
    # ---------------------------
    def draw_ball_highlight(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = max((x2 - x1) // 2, (y2 - y1) // 2)

        pulse = int(np.sin(cv2.getTickCount() * 0.00035) * 6 + 12)
        for i in range(4):
            overlay = frame.copy()
            r = radius + pulse + i * 3
            alpha = max(0.28 - i * 0.07, 0.05)
            cv2.circle(overlay, (cx, cy), r, (0, 255, 255), 2)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.circle(frame, (cx, cy), radius, (0, 255, 255), 2)
        self.draw_text_with_shadow(frame, "BALL", (cx - 18, cy - radius - 8),
                                   font_scale=0.45, color=(0, 255, 255), thickness=1)
        return frame

    # ---------------------------
    # Main drawing
    # ---------------------------
    def draw_enhanced_frame(self, frame, detections, team_assignments):
        """
        Draw all entities (players, goalkeepers, referees, ball)
        using team classification results.
        """
        # Trails first
        for det in detections:
            gid = det.get('global_id')
            cls_name = det.get('class', 'player')
            tracked = det.get('tracked', False)

            if tracked and gid is not None and gid != -1 and cls_name in ['player', 'goalkeeper']:
                team_id = team_assignments.get(gid, -1)
                is_gk = (cls_name == 'goalkeeper')
                frame = self.draw_track_trail(frame, gid, team_id, is_gk)

        # Entities next
        for det in detections:
            bbox = det.get('bbox')
            if not bbox:
                continue

            gid = det.get('global_id')
            cls_name = det.get('class', 'player')
            conf = det.get('confidence', 1.0)
            tracked = det.get('tracked', False)

            if cls_name == 'ball':
                frame = self.draw_ball_highlight(frame, bbox)

            elif cls_name == 'goalkeeper':
                tid = team_assignments.get(gid, -1)
                frame = self.draw_player_ellipse(frame, bbox, gid, tid, conf, is_goalkeeper=True)

            elif cls_name == 'player':
                if tracked and gid is not None:
                    tid = team_assignments.get(gid, -1)
                    frame = self.draw_player_ellipse(frame, bbox, gid, tid, conf, is_goalkeeper=False)
                else:
                    # Untracked detection → thin gray ellipse
                    x1, y1, x2, y2 = map(int, bbox)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    axes = (max(5, (x2 - x1) // 2), max(5, (y2 - y1) // 2))
                    cv2.ellipse(frame, center, axes, 0, 0, 360, (180, 180, 180), 1)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            elif cls_name == 'referee':
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.referee_color, 2)
                self.draw_text_with_shadow(frame, "REF", (x1, y1 - 6),
                                           font_scale=0.5, color=self.referee_color, thickness=1)

        # Finally draw team legend
        frame = self.draw_team_legend(frame)
        return frame

    # ---------------------------
    # Compact Team Legend
    # ---------------------------
    def draw_team_legend(self, frame):
        h, w = frame.shape[:2]
        legend_x = w - 130
        legend_y = 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (legend_x - 8, legend_y - 8),
                      (w - 10, legend_y + 70), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        y_offset = 0
        for team_id, color in self.team_classifier.team_colors.items():
            if team_id == -1:
                continue
            y_pos = legend_y + y_offset
            cv2.circle(frame, (legend_x, y_pos), 6, color, -1)
            cv2.circle(frame, (legend_x, y_pos), 6, (255, 255, 255), 1)
            text = f"Team {chr(65 + team_id)}"
            self.draw_text_with_shadow(frame, text,
                                       (legend_x + 15, y_pos + 4),
                                       font_scale=0.45, color=(255, 255, 255), thickness=1)
            y_offset += 20

        # GK legend
        y_pos = legend_y + y_offset
        cv2.circle(frame, (legend_x, y_pos), 6, self.goalkeeper_color, -1)
        cv2.circle(frame, (legend_x, y_pos), 6, (255, 255, 255), 1)
        self.draw_text_with_shadow(frame, "GK",
                                   (legend_x + 15, y_pos + 4),
                                   font_scale=0.45, color=(255, 255, 255), thickness=1)
        return frame