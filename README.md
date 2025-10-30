# ⚽ Football Tracking and Analytics System

This project is a **comprehensive football analytics pipeline** that combines **object detection**, **multi-object tracking**, **team classification**, and **advanced tactical analytics**.  
It leverages **YOLOv11**, **BoT-SORT tracking**, and **custom analytical modules** to analyze matches and extract meaningful statistics, visualizations, and heatmaps for both teams and individual players.

---

## 🚀 Features Overview

- **Detection:** Detects all players, referees, goalkeepers, and the ball using YOLOv11.  
- **Tracking:** Tracks detected objects over time with BoT-SORT.  
- **Team Classification:** Classifies players into their teams using either:
  - 🟥 `ColorTeamClassifier` (fast, efficient, works well)
  - 🟦 `SigLIPTeamClassifier` (more robust, slower, requires higher computation)
- **Analytics Export:** Generates structured CSV files, plots, and heatmaps for team and player performance.
- **Customizable Configuration:** Modify `config.py` and `configs/botsort.yaml` to tune behavior per match/video.

---

## 🧩 Core Modules Description

| Module | Description |
|--------|--------------|
| **`src/main_enhanced.py`** | The main execution script — integrates detection, tracking, team classification, and analytics generation. |
| **`src/detection.py`** | Handles YOLOv11 object detection for players, referees, goalkeepers, and the ball. |
| **`src/botsort_tracker.py`** | Implements BoT-SORT multi-object tracker to maintain consistent IDs across frames. |
| **`src/team_classifier.py`** | Uses color features from uniforms to classify players into teams (fast). |
| **`src/siglip_team_classifier.py`** | Uses a SigLIP-based visual-text model for more accurate but slower team classification. |
| **`src/feature_extractor.py`** | Extracts color and spatial features for classification and analytics. |
| **`src/id_manager_simple.py`** | Manages player IDs and mappings between frames. |
| **`src/analytics.py`** | Processes positional and performance data to generate advanced statistics. |
| **`src/export_manager.py`** | Exports analytical data, plots, and heatmaps to the `analytics_export/` folder. |
| **`src/visualizer.py`** | Handles match video overlays, bounding boxes, and visualization of player movements. |
| **`src/utils.py`** | Helper functions for preprocessing, coordinate conversion, and drawing. |
| **`src/config.py`** | Configurable parameters — adjust paths, thresholds, frame limits, etc. |
| **`configs/botsort.yaml`** | BoT-SORT configuration for tracker thresholds and ReID settings. |

---

## 🖼️ Example Analytics Results

Below are sample outputs from the analytics stage.  
These are automatically generated under `analytics_export/heatmaps/`.

| Team 0 Heatmap | Team 1 Heatmap |
|----------------|----------------|
| ![Team 0 Heatmap](analytics_export/heatmaps/team_0_heatmap.png) | ![Team 1 Heatmap](analytics_export/heatmaps/team_1_heatmap.png) |

---

## 🧠 How to Run

```bash
python main_enhanced.py `
  --video "path_to_your_match_video.mp4" `
  --out "output/tracked.mp4" `
  --model "models/best (4).pt" `
  --tracker-config "configs/botsort.yaml" `
  --no-siglip


💡 If you want to use the SigLIP Team Classifier, remove the --no-siglip flag.
You can adjust thresholds and parameters in config.py or configs/botsort.yaml depending on your video or environment.

📊 Generated Results

After successful execution, the following will be generated:

analytics_export/
│
├── data/
│   ├── id_mappings.json
│   ├── player_metrics.csv
│   └── player_positions.csv
│   ├── output/
      ├── tracked
├── heatmaps/
│   ├── team_0_heatmap.png
│   └── team_1_heatmap.png
│
├── player_heatmaps/
│   ├── player_1_heatmap.png
│   └── ...
│
├── performance_plots/
│   ├── sprint_count.png
│   ├── speed_comparison.png
│   ├── team_comparison_dashboard.png
│   └── ...
│
└── reports/
    └── analysis_summary.txt

📦 Installation
git clone https://github.com/YoussefEsmael/Football-Tracking-and-Analytics-System.git
cd Football-Tracking-and-Analytics-System
pip install -r requirements.txt

🔗 Model Access

The trained YOLOv11 model used in this project can be accessed via the provided Google Drive link (add yours here):

Download Model from Google Drive

⚙️ Limitations

Re-identification:

ReID occasionally assigns new IDs if a player disappears and reappears after several frames.

Fine-tuning this part can lead to even more stable analytics and richer statistics.

Occlusion Handling:

Occlusions between players are handled very well — team classification remains accurate.

In rare cases, one of the occluded players might be given a new ID.

🏁 Summary

This system provides an end-to-end football analytics pipeline that detects, tracks, classifies, and analyzes football match footage — generating insightful visual and numerical outputs that can assist analysts, coaches, and AI researchers in sports analytics.

🧾 Author

Youssef Esmael
📍 Egypt
📧 ismmailmuhamed@gmail.com
