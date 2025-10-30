# ⚽ Advanced Football Tracking & Analytics System

A comprehensive computer vision pipeline for automatic football match analysis using state-of-the-art deep learning models. This system tracks players, classifies teams, and generates detailed performance analytics from broadcast video footage.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

### Core Capabilities
- **YOLOv11 Object Detection** - Detects players, goalkeepers, referees, and ball
- **BoT-SORT Multi-Object Tracking** - Maintains consistent player identities with built-in ReID
- **SigLIP + UMAP Team Classification** - Vision-language model for robust jersey color recognition
- **Global ID Management** - Persistent player tracking across occlusions and camera cuts
- **Advanced Analytics Engine** - Comprehensive performance metrics and tactical insights

### Analytics Outputs
- 📊 **Performance Metrics**: Distance covered, speed analysis, sprint detection
- 🎯 **Passing Analysis**: Pass networks, accuracy, completion rates
- 🗺️ **Heatmaps**: Individual player movement, team positioning, sprint zones
- ⚡ **Tactical Events**: Possession tracking, transitions, pressing intensity
- 🏟️ **Zone Control**: Field occupation analysis with 3x3 grid system
- 📈 **Visualizations**: Team comparison dashboards, performance plots

## 🎥 Results

### Sample Output Video
*[Your output video showcasing tracked players with colored ellipses, trails, and team assignments]*

![Sample Frame]("C:\Users\yossef\Desktop\football is life\output-no-siglip.mp4")

### Analytics Dashboard
*Team comparison showing distance, passes, and speed metrics*

![Team Dashboard](path/to/team_comparison_dashboard.png)

### Player Heatmaps
*Individual movement patterns across the pitch*

| Player Heatmap | Sprint Zones |
|:---:|:---:|
| ![Heatmap](path/to/player_heatmap.png) | ![Sprints](path/to/sprint_zones.png) |

### Performance Metrics
![Distance Comparison](path/to/distance_per_player.png)
![Speed Analysis](path/to/speed_comparison.png)

## 📁 Project Structure

```
football_tracking/
├── main.py                          # Main pipeline script
├── config.py                        # Configuration parameters
├── botsort_tracker.py              # BoT-SORT integration
├── siglip_team_classifier.py       # SigLIP-based team classification
├── team_classifier.py              # Fallback color-based classifier
├── id_manager_simple.py            # Global ID management
├── visualizer.py                   # Enhanced visualization
├── analytics.py                    # Analytics engine
├── export_manager.py               # Data export & plots
├── utils.py                        # Utility functions
├── detection.py                    # Detection manager
├── feature_extractor.py            # Feature extraction
├── configs/
│   └── botsort.yaml                # BoT-SORT configuration
├── models/
│   └── best.pt                     # YOLOv11 model (see below)
├── analytics_export/               # Generated outputs
│   ├── data/                       # CSV exports
│   │   ├── player_positions.csv
│   │   ├── player_metrics.csv
│   │   ├── ball_positions.csv
│   │   └── pass_events.csv
│   ├── player_heatmaps/           # Individual heatmaps
│   ├── performance_plots/         # Performance visualizations
│   └── reports/                   # Summary reports
└── README.md
```

**Note**: The following analytics are **currently disabled** due to accuracy limitations:
- `momentum/` - Momentum analysis
- `pass_networks/` - Pass network graphs  
- `possession/` - Possession timeline
- `transitions/` - Turnover analysis
- `zone_control/` - Field zone heatmaps

These features are experimental and may produce inconsistent results in the current version.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for real-time processing)
- 8GB+ RAM

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/football-tracking.git
cd football-tracking
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Core dependencies** (see `requirements.txt` for full list):
- `ultralytics` - YOLOv11 detection & BoT-SORT tracking
- `transformers` + `torch` - SigLIP team classification
- `umap-learn` - Dimensionality reduction for clustering
- `opencv-python` - Video processing
- `scikit-learn` - Clustering algorithms
- `pandas` + `matplotlib` - Analytics & visualization

### Step 4: Download YOLOv11 Model
The custom-trained YOLOv11 model is **not included** in this repository due to file size.

📥 **Download Link**: [YOLOv11 Football Model on Google Drive](https://drive.google.com/your-model-link)

After downloading:
1. Place `best.pt` in the `models/` directory
2. Update `config.py` if using a different path:
   ```python
   YOLO_MODEL_PATH = "path/to/your/model.pt"
   ```

## 📖 Usage

### Basic Usage
```bash
python main.py --video path/to/match.mp4 --out output/tracked_match.mp4
```

### Advanced Options
```bash
python main.py \
    --video input/match.mp4 \
    --out output/result.mp4 \
    --max-frames 1000 \
    --tracker-config configs/botsort.yaml \
    --no-siglip  # Use color-based classification instead
```

### Command-Line Arguments
- `--video` / `-v`: Path to input video (required)
- `--out` / `-o`: Path to output video (default: `output_tracked.mp4`)
- `--max-frames`: Maximum frames to process (0 = all frames)
- `--tracker-config`: Custom BoT-SORT YAML config
- `--no-siglip`: Disable SigLIP (use faster color-based team classification)

### Configuration

#### `config.py` - Main Settings
Customize these parameters based on your use case:

```python
# Detection
CONFIDENCE_THRESHOLD = 0.35          # Lower = more detections, higher = fewer false positives

# Global ID Management  
REID_SPATIAL_THRESHOLD = 250         # Max pixel movement between frames
REID_TEMPORAL_WINDOW = 60            # Frames to remember lost tracks

# Team Classification
TEAM_WARMUP_FRAMES = 25              # Frames before clustering teams
TEAM_CONFIDENCE_THRESHOLD = 0.4      # Minimum confidence for team assignment

# Analytics
HIGH_SPEED_THRESHOLD = 15.0          # km/h for high-intensity running
SPRINT_THRESHOLD = 20.0              # km/h for sprint detection

# Auto-adjust for high FPS videos
AUTO_ADJUST_FOR_FPS = True           # Scale thresholds based on video FPS
```

#### `configs/botsort.yaml` - Tracker Settings
Fine-tune tracking behavior:

```yaml
tracker_type: botsort

# Detection thresholds
track_high_thresh: 0.6               # High confidence threshold
track_low_thresh: 0.1                # Low confidence threshold  
new_track_thresh: 0.7                # Threshold for new tracks

# Track management
track_buffer: 30                     # Frames to keep lost tracks (adjust for FPS)
match_thresh: 0.8                    # IoU threshold for matching

# ReID
with_reid: true                      # Enable appearance-based ReID
proximity_thresh: 0.5                # Proximity threshold
appearance_thresh: 0.25              # Appearance similarity threshold

# Camera motion compensation
gmc_method: sparseOptFlow            # Options: sparseOptFlow, orb, ecc, none
```

**For 50+ FPS videos**, increase `track_buffer` proportionally (e.g., 60 for 50 FPS).

## ⚠️ Known Limitations

### 1. **Re-Identification (ReID) Challenges**
- **Issue**: When a player temporarily leaves the frame (e.g., goes off-screen, behind referee) and reappears, they may be assigned a **new ID** instead of recovering their original ID
- **Impact**: Affects long-term tracking consistency and per-player statistics
- **Mitigation**: 
  - Increase `REID_TEMPORAL_WINDOW` in `config.py` for longer memory
  - Increase `track_buffer` in `botsort.yaml` 
  - Use higher FPS videos for better temporal continuity
- **Status**: Inherent limitation of current BoT-SORT + spatial-temporal matching approach

### 2. **Occlusion Handling**
- **Issue**: When players overlap heavily (e.g., during tackles, set pieces), the occluded player may:
  - Temporarily lose tracking
  - Be assigned a new ID when visible again
- **Impact**: ID fragmentation in crowded scenes
- **Works Well**: Short-term occlusions (1-3 frames) are handled correctly
- **Fails**: Extended occlusions (5+ frames) with significant player movement
- **Mitigation**: Team classification remains accurate during occlusion, only ID consistency is affected

### 3. **Team Classification Edge Cases**
- **Issue**: Players with similar jersey colors (e.g., dark blue vs black) may be misclassified
- **Works Well**: Distinct team colors (red vs blue, white vs dark)
- **Mitigation**: SigLIP model provides better robustness than traditional color histograms

### 4. **Camera Angle Dependency**
- **Best Results**: Broadcast-style tactical camera (side view, elevated)
- **Limitations**: Close-ups, behind-goal views, and rapidly moving cameras reduce accuracy

### 5. **Disabled Analytics**
The following outputs are **experimental** and disabled due to accuracy concerns:
- Pass network detection (requires ball tracking improvements)
- Possession analysis (needs refined ball proximity logic)
- Zone control heatmaps (experimental)
- Transition/momentum analysis (experimental)

These features exist in the codebase but are not actively maintained.

## 🎯 Best Practices

### For Optimal Results:
1. **Video Quality**: Use 720p+ resolution, 25+ FPS
2. **Camera Angle**: Tactical/broadcast view covering most of the pitch
3. **Lighting**: Well-lit matches (avoid shadows, night games with poor lighting)
4. **Team Colors**: Distinct jersey colors produce best classification
5. **Configuration**: Adjust thresholds in `config.py` for your specific video characteristics

### For High FPS Videos (50+ FPS):
- Set `AUTO_ADJUST_FOR_FPS = True` in `config.py`
- Or manually increase:
  - `REID_SPATIAL_THRESHOLD` → 400+
  - `REID_TEMPORAL_WINDOW` → 120+
  - `track_buffer` in `botsort.yaml` → 60+

## 📊 Output Files

After processing, find results in `analytics_export/`:

### Data Exports (CSV)
- `data/player_positions.csv` - Frame-by-frame player positions
- `data/player_metrics.csv` - Per-player performance summary
- `data/ball_positions.csv` - Ball tracking data
- `data/pass_events.csv` - Detected passing events

### Visualizations (PNG)
- `player_heatmaps/` - Individual movement heatmaps
- `performance_plots/distance_per_player.png` - Distance comparison
- `performance_plots/speed_comparison.png` - Speed metrics
- `performance_plots/team_comparison_dashboard.png` - Overall dashboard

### Summary
- `reports/analysis_summary.txt` - Text summary of results

## 🛠️ Troubleshooting

### "Cannot open video writer"
- **Solution**: Install alternative codecs or change output format:
  ```bash
  pip install opencv-contrib-python
  ```
  Or try different extension: `--out output.avi`

### "CUDA out of memory"
- **Solution**: Reduce batch size or use CPU:
  ```python
  # In config.py
  ENABLE_GPU = False
  ```

### "SigLIP failed to load"
- **Solution**: System will automatically fallback to color-based classification
- For GPU acceleration: Ensure PyTorch CUDA is installed correctly

### Poor tracking quality
1. Check video FPS and adjust `track_buffer` accordingly
2. Increase `CONFIDENCE_THRESHOLD` to reduce false detections
3. Verify YOLOv11 model is loaded correctly
4. Try processing a shorter clip first (use `--max-frames 500`)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Improved ReID model training for football-specific scenarios
- [ ] Enhanced occlusion handling algorithms
- [ ] Ball trajectory prediction
- [ ] Automatic highlight detection
- [ ] Real-time processing optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics YOLOv11** - Object detection framework
- **BoT-SORT** - Multi-object tracking algorithm
- **SigLIP** - Vision-language model by Google
- **UMAP** - Dimensionality reduction

## 📞 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐

