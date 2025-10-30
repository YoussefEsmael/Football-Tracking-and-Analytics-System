"""
Enhanced ExportManager with valuable football insights
Aligned with team_id (0/1) and global_id system
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns


class ExportManager:
    """Manages export of tracking data, metrics, and visual analytics"""

    def __init__(self, output_dir="analytics_export"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["data", "reports", "visualizations", "player_heatmaps", 
                   "performance_plots", "pass_networks", "sprint_heatmaps", 
                   "momentum", "possession", "transitions", "zone_control"]
        for sub in subdirs:
            (self.output_dir / sub).mkdir(exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'

    def _valid_players(self, team_assignments):
        """Return only global IDs with team 0 or 1"""
        return {pid: team for pid, team in team_assignments.items() if team in [0, 1]}

    # ============================================================================
    # CSV EXPORTS
    # ============================================================================
    
    def export_tracking_data(self, analytics, team_assignments):
        """Export raw tracking data"""
        print("[Export] Exporting tracking data...")
        valid_ids = self._valid_players(team_assignments)

        try:
            # Player positions
            rows = []
            for gid, pos_list in analytics.player_positions.items():
                if gid not in valid_ids:
                    continue
                for p in pos_list:
                    rows.append({
                        "global_id": gid,
                        "team_id": valid_ids[gid],
                        "frame": p.get("frame", 0),
                        "timestamp": p.get("timestamp", 0.0),
                        "x": p.get("x", 0.0),
                        "y": p.get("y", 0.0),
                        "speed": p.get("speed", 0.0)
                    })
            if rows:
                pd.DataFrame(rows).to_csv(
                    self.output_dir / "data" / "player_positions.csv", index=False)

            # Ball positions
            if analytics.ball_positions:
                pd.DataFrame(analytics.ball_positions).to_csv(
                    self.output_dir / "data" / "ball_positions.csv", index=False)

            # Pass events
            if analytics.pass_events:
                pass_rows = []
                for p in analytics.pass_events:
                    if (p['passer_id'] in valid_ids and 
                        p['receiver_id'] in valid_ids):
                        pass_rows.append(p)
                if pass_rows:
                    pd.DataFrame(pass_rows).to_csv(
                        self.output_dir / "data" / "pass_events.csv", index=False)

            # Possession windows
            if analytics.possession_windows:
                pd.DataFrame(analytics.possession_windows).to_csv(
                    self.output_dir / "data" / "possession_windows.csv", index=False)

            # Transitions
            if analytics.transitions:
                pd.DataFrame(analytics.transitions).to_csv(
                    self.output_dir / "data" / "transitions.csv", index=False)

            # Pressing events
            if analytics.pressing_events:
                pd.DataFrame(analytics.pressing_events).to_csv(
                    self.output_dir / "data" / "pressing_events.csv", index=False)

            print(f"[Export] Tracking data saved to {self.output_dir / 'data'}")

        except Exception as e:
            print(f"[Export] ERROR: {e}")

    def export_performance_metrics(self, analytics, team_assignments):
        """Export per-player performance metrics"""
        print("[Export] Exporting performance metrics...")
        valid_ids = self._valid_players(team_assignments)

        try:
            rows = []
            for gid, metrics in analytics.player_metrics.items():
                if gid not in valid_ids:
                    continue
                row = {"global_id": gid, "team_id": valid_ids[gid]}
                row.update(metrics)
                rows.append(row)
            
            if rows:
                pd.DataFrame(rows).to_csv(
                    self.output_dir / "data" / "player_metrics.csv", index=False)
                print(f"[Export] Metrics for {len(rows)} players saved")
        except Exception as e:
            print(f"[Export] ERROR: {e}")

    # ============================================================================
    # HEATMAPS
    # ============================================================================
    
    def generate_individual_player_heatmaps(self, analytics, team_assignments, frame_shape):
        """Generate heatmap for each player"""
        h, w = frame_shape
        out_dir = self.output_dir / "player_heatmaps"
        print("[ExportManager] Generating individual player heatmaps...")

        valid_ids = self._valid_players(team_assignments)

        for gid, positions in analytics.player_positions.items():
            if gid not in valid_ids or len(positions) < 10:
                continue

            arr = np.array([(p['x'], p['y']) for p in positions])
            heatmap, xedges, yedges = np.histogram2d(
                arr[:, 0], arr[:, 1], bins=60, density=True)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                          cmap='YlOrRd', alpha=0.8)
            ax.set_title(f"Player {gid} (Team {valid_ids[gid]}) Movement Heatmap", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("X Position (pixels)", fontsize=12)
            ax.set_ylabel("Y Position (pixels)", fontsize=12)
            fig.colorbar(im, ax=ax, label='Density')
            
            fig.savefig(out_dir / f"player_{gid}_heatmap.png", 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)

    def generate_sprint_heatmaps(self, analytics, team_assignments, frame_shape, 
                                 speed_threshold=20.0):
        """Generate heatmap showing only high-speed positions"""
        out_dir = self.output_dir / "sprint_heatmaps"
        print("[ExportManager] Generating sprint heatmaps...")

        valid_ids = self._valid_players(team_assignments)

        for gid, positions in analytics.player_positions.items():
            if gid not in valid_ids:
                continue

            sprint_positions = [p for p in positions if p.get("speed", 0) >= speed_threshold]
            if len(sprint_positions) < 5:
                continue

            arr = np.array([(p['x'], p['y']) for p in sprint_positions])
            heatmap, xedges, yedges = np.histogram2d(
                arr[:, 0], arr[:, 1], bins=40, density=True)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                          cmap='Blues', alpha=0.8)
            ax.set_title(f"Player {gid} (Team {valid_ids[gid]}) Sprint Zones", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("X Position", fontsize=12)
            ax.set_ylabel("Y Position", fontsize=12)
            fig.colorbar(im, ax=ax, label='Sprint Density')
            
            fig.savefig(out_dir / f"player_{gid}_sprint_heatmap.png", 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)

    # ============================================================================
    # PERFORMANCE VISUALIZATIONS
    # ============================================================================
    
    def generate_performance_visualizations(self, analytics, team_assignments):
        """Generate comprehensive performance charts"""
        out_dir = self.output_dir / "performance_plots"
        print("[ExportManager] Generating performance visualizations...")

        try:
            valid_ids = self._valid_players(team_assignments)
            rows = []
            for gid, metrics in analytics.player_metrics.items():
                if gid not in valid_ids:
                    continue
                row = {"player_id": gid, "team_id": valid_ids[gid]}
                row.update(metrics)
                rows.append(row)
            
            if not rows:
                print("[ExportManager] No valid players to visualize")
                return
            
            df = pd.DataFrame(rows)
            
            # 1. Distance comparison
            if "total_distance" in df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                team_colors = {0: '#FF3333', 1: '#3333FF'}
                for team_id in [0, 1]:
                    team_df = df[df['team_id'] == team_id].sort_values('total_distance', ascending=False)
                    ax.barh(team_df['player_id'].astype(str), team_df['total_distance'], 
                           label=f'Team {team_id}', color=team_colors[team_id], alpha=0.7)
                
                ax.set_xlabel("Distance Covered (meters)", fontsize=12)
                ax.set_ylabel("Player ID", fontsize=12)
                ax.set_title("Total Distance Covered by Player", fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / "distance_per_player.png", dpi=150)
                plt.close()

            # 2. Speed metrics comparison
            if "max_speed" in df.columns and "avg_speed" in df.columns:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Max speed
                for team_id in [0, 1]:
                    team_df = df[df['team_id'] == team_id]
                    ax1.scatter(team_df['player_id'], team_df['max_speed'], 
                               label=f'Team {team_id}', s=100, alpha=0.7)
                ax1.set_xlabel("Player ID", fontsize=12)
                ax1.set_ylabel("Max Speed (km/h)", fontsize=12)
                ax1.set_title("Maximum Speed by Player", fontsize=13, fontweight='bold')
                ax1.legend()
                ax1.grid(alpha=0.3)
                
                # Average speed
                for team_id in [0, 1]:
                    team_df = df[df['team_id'] == team_id]
                    ax2.scatter(team_df['player_id'], team_df['avg_speed'], 
                               label=f'Team {team_id}', s=100, alpha=0.7)
                ax2.set_xlabel("Player ID", fontsize=12)
                ax2.set_ylabel("Average Speed (km/h)", fontsize=12)
                ax2.set_title("Average Speed by Player", fontsize=13, fontweight='bold')
                ax2.legend()
                ax2.grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(out_dir / "speed_comparison.png", dpi=150)
                plt.close()

            # 3. Pass accuracy and volume
            if "pass_accuracy" in df.columns and "passes_made" in df.columns:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for team_id in [0, 1]:
                    team_df = df[df['team_id'] == team_id]
                    ax.scatter(team_df['passes_made'], team_df['pass_accuracy'], 
                              s=200, alpha=0.6, label=f'Team {team_id}')
                    
                    # Add player labels
                    for _, row in team_df.iterrows():
                        ax.annotate(str(row['player_id']), 
                                   (row['passes_made'], row['pass_accuracy']),
                                   fontsize=9, ha='center')
                
                ax.set_xlabel("Total Passes Made", fontsize=12)
                ax.set_ylabel("Pass Accuracy (%)", fontsize=12)
                ax.set_title("Pass Volume vs Accuracy", fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / "pass_accuracy_vs_volume.png", dpi=150)
                plt.close()

            # 4. Sprint count comparison
            if "sprints" in df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                team_colors = {0: '#FF3333', 1: '#3333FF'}
                
                x = np.arange(len(df))
                width = 0.35
                
                team0 = df[df['team_id'] == 0]
                team1 = df[df['team_id'] == 1]
                
                ax.bar(team0['player_id'].astype(str), team0['sprints'], 
                       color=team_colors[0], alpha=0.7, label='Team 0')
                ax.bar(team1['player_id'].astype(str), team1['sprints'], 
                       color=team_colors[1], alpha=0.7, label='Team 1')
                
                ax.set_xlabel("Player ID", fontsize=12)
                ax.set_ylabel("Number of Sprints", fontsize=12)
                ax.set_title("Sprint Count by Player", fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / "sprint_count.png", dpi=150)
                plt.close()

            # 5. High intensity running
            if "high_intensity_distance" in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                df_sorted = df.sort_values('high_intensity_distance', ascending=True)
                colors = [team_colors[t] for t in df_sorted['team_id']]
                
                ax.barh(df_sorted['player_id'].astype(str), 
                       df_sorted['high_intensity_distance'],
                       color=colors, alpha=0.7)
                ax.set_xlabel("High Intensity Distance (meters)", fontsize=12)
                ax.set_ylabel("Player ID", fontsize=12)
                ax.set_title("High Intensity Running Distance", fontsize=14, fontweight='bold')
                
                # Custom legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=team_colors[0], alpha=0.7, label='Team 0'),
                                  Patch(facecolor=team_colors[1], alpha=0.7, label='Team 1')]
                ax.legend(handles=legend_elements)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / "high_intensity_distance.png", dpi=150)
                plt.close()

            # 6. Team comparison dashboard
            self._generate_team_comparison(df, out_dir)
            
            print(f"[ExportManager] Performance visualizations saved to {out_dir}")

        except Exception as e:
            print(f"[ExportManager] ERROR: {e}")
            import traceback
            traceback.print_exc()

    def _generate_team_comparison(self, df, out_dir):
        """Generate comprehensive team comparison dashboard"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        team_colors = {0: '#FF3333', 1: '#3333FF'}
        
        # Aggregate team stats
        team_stats = df.groupby('team_id').agg({
            'total_distance': 'sum',
            'passes_made': 'sum',
            'pass_accuracy': 'mean',
            'sprints': 'sum',
            'max_speed': 'max',
            'avg_speed': 'mean',
            'high_intensity_distance': 'sum',
            'touches': 'sum'
        }).reset_index()
        
        # 1. Total distance
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(team_stats['team_id'].astype(str), team_stats['total_distance'],
               color=[team_colors[t] for t in team_stats['team_id']], alpha=0.7)
        ax1.set_title('Total Distance', fontweight='bold')
        ax1.set_ylabel('Meters')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Passes made
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(team_stats['team_id'].astype(str), team_stats['passes_made'],
               color=[team_colors[t] for t in team_stats['team_id']], alpha=0.7)
        ax2.set_title('Total Passes', fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Pass accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(team_stats['team_id'].astype(str), team_stats['pass_accuracy'],
               color=[team_colors[t] for t in team_stats['team_id']], alpha=0.7)
        ax3.set_title('Pass Accuracy', fontweight='bold')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_ylim([0, 100])
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Sprint count
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(team_stats['team_id'].astype(str), team_stats['sprints'],
               color=[team_colors[t] for t in team_stats['team_id']], alpha=0.7)
        ax4.set_title('Total Sprints', fontweight='bold')
        ax4.set_ylabel('Count')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Max speed
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.bar(team_stats['team_id'].astype(str), team_stats['max_speed'],
               color=[team_colors[t] for t in team_stats['team_id']], alpha=0.7)
        ax5.set_title('Top Speed', fontweight='bold')
        ax5.set_ylabel('km/h')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. High intensity distance
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.bar(team_stats['team_id'].astype(str), team_stats['high_intensity_distance'],
               color=[team_colors[t] for t in team_stats['team_id']], alpha=0.7)
        ax6.set_title('High Intensity Distance', fontweight='bold')
        ax6.set_ylabel('Meters')
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Player count
        ax7 = fig.add_subplot(gs[2, 0])
        player_counts = df.groupby('team_id').size()
        ax7.bar(player_counts.index.astype(str), player_counts.values,
               color=[team_colors[t] for t in player_counts.index], alpha=0.7)
        ax7.set_title('Players Tracked', fontweight='bold')
        ax7.set_ylabel('Count')
        ax7.grid(axis='y', alpha=0.3)
        
        # 8. Touches
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.bar(team_stats['team_id'].astype(str), team_stats['touches'],
               color=[team_colors[t] for t in team_stats['team_id']], alpha=0.7)
        ax8.set_title('Total Touches', fontweight='bold')
        ax8.set_ylabel('Count')
        ax8.grid(axis='y', alpha=0.3)
        
        # 9. Summary text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        summary_text = "TEAM COMPARISON\n\n"
        for _, row in team_stats.iterrows():
            team_id = int(row['team_id'])
            summary_text += f"Team {team_id}:\n"
            summary_text += f"  Distance: {row['total_distance']:.1f}m\n"
            summary_text += f"  Passes: {row['passes_made']:.0f}\n"
            summary_text += f"  Accuracy: {row['pass_accuracy']:.1f}%\n\n"
        ax9.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                fontfamily='monospace')
        
        fig.suptitle('Team Performance Comparison Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(out_dir / "team_comparison_dashboard.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ============================================================================
    # POSSESSION & TRANSITIONS
    # ============================================================================
    
    def generate_possession_analysis(self, analytics, team_assignments):
        """Generate possession timeline and statistics"""
        out_dir = self.output_dir / "possession"
        print("[ExportManager] Generating possession analysis...")
        
        if not analytics.possession_windows:
            print("[ExportManager] No possession data available")
            return
        
        # Calculate possession percentages
        total_time = sum(w['duration'] for w in analytics.possession_windows)
        team_possession = {0: 0, 1: 0}
        
        for window in analytics.possession_windows:
            team_id = window['team_id']
            if team_id in [0, 1]:
                team_possession[team_id] += window['duration']
        
        # Possession pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        pct_0 = (team_possession[0] / total_time * 100) if total_time > 0 else 0
        pct_1 = (team_possession[1] / total_time * 100) if total_time > 0 else 0
        
        colors = ['#FF3333', '#3333FF']
        ax1.pie([pct_0, pct_1], labels=['Team 0', 'Team 1'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Possession Distribution', fontsize=14, fontweight='bold')
        
        # Possession timeline
        timeline_data = []
        for w in analytics.possession_windows:
            if w['team_id'] in [0, 1]:
                timeline_data.append({
                    'start': w['start_frame'],
                    'end': w['end_frame'],
                    'team': w['team_id']
                })
        
        if timeline_data:
            for entry in timeline_data[:100]:  # Limit to first 100 for visibility
                color = colors[entry['team']]
                ax2.barh(0, entry['end'] - entry['start'], 
                        left=entry['start'], color=color, alpha=0.7)
            
            ax2.set_title('Possession Timeline', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Frame Number', fontsize=12)
            ax2.set_yticks([])
            ax2.legend(['Team 0', 'Team 1'])
        
        plt.tight_layout()
        plt.savefig(out_dir / "possession_analysis.png", dpi=150)
        plt.close()
        
        print(f"[ExportManager] Team 0 possession: {pct_0:.1f}%")
        print(f"[ExportManager] Team 1 possession: {pct_1:.1f}%")

    def generate_transition_analysis(self, analytics):
        """Analyze possession transitions and turnovers"""
        out_dir = self.output_dir / "transitions"
        print("[ExportManager] Generating transition analysis...")
        
        if not analytics.transitions:
            print("[ExportManager] No transition data available")
            return
        
        df_trans = pd.DataFrame(analytics.transitions)
        
        # Count transitions by team
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Turnovers given away
        turnovers_from = df_trans['from_team'].value_counts()
        ax1.bar(turnovers_from.index.astype(str), turnovers_from.values,
               color=['#FF3333', '#3333FF'], alpha=0.7)
        ax1.set_title('Turnovers Given Away', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Team', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Turnovers won
        turnovers_to = df_trans['to_team'].value_counts()
        ax2.bar(turnovers_to.index.astype(str), turnovers_to.values,
               color=['#FF3333', '#3333FF'], alpha=0.7)
        ax2.set_title('Turnovers Won', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Team', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / "transition_analysis.png", dpi=150)
        plt.close()

    def save_all(self):
        """Finalize all exports and generate summary report"""
        print("[ExportManager] Generating summary report...")
        
        report_path = self.output_dir / "reports" / "analysis_summary.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FOOTBALL TRACKING ANALYSIS SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            f.write("Available exports:\n")
            f.write("  - Player tracking data (CSV)\n")
            f.write("  - Performance metrics (CSV)\n")
            f.write("  - Pass networks (PNG)\n")
            f.write("  - Heatmaps (PNG)\n")
            f.write("  - Performance visualizations (PNG)\n")
            f.write("  - Possession analysis (PNG)\n")
            f.write("  - Transition analysis (PNG)\n")
        
        print(f"[ExportManager] Summary saved to {report_path}")
        print("[ExportManager] All exports complete!")