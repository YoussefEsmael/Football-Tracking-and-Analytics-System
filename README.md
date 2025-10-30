▶️ Usage

Run the full analytics pipeline (detection → tracking → team classification → analytics export):

python src/main_enhanced.py --video "path/to/match.mp4"


Example:

python src/main_enhanced.py \
  --video "input/match.mp4" \
  --out "output/processed_match.mp4" \
  --model "models/best (4).pt" \
  --tracker-config "configs/botsort.yaml" \
  --no-siglip


💡 Remove --no-siglip if you want to use the SigLIP team classifier.

🎥 Output Folder

After execution, an output folder will be generated containing the processed match video.

output/
 └── processed_match.mp4        # Video with detection, tracking, and team overlays

📊 Analytics Export Folder

Analytics are automatically saved in the following structure:

analytics_export/
 ├── heatmaps/                     # Team and player positional heatmaps
 ├── performance_plots/            # Sprint, speed, and comparative plots
 ├── data/                         # CSV files for player positions & metrics
 ├── player_heatmaps/              # Individual player activity maps
 └── reports/
     └── analysis_summary.txt      # Match summary report

🔥 Example Heatmaps

Visualizing player movement and positional density across the field.

Team 0 Heatmap	Team 1 Heatmap

	
📈 Performance Plots

Illustrating sprint counts, speed metrics, and team comparisons.

Sprint Count	Speed Comparison

	
Team Comparison Dashboard	Player Performance

	
🧩 Configuration

You can modify detection, tracking, and classification parameters in the following files:

src/config.py

Contains global settings for:

Frame rate

Detection thresholds

Model paths

Output paths

configs/botsort.yaml

Adjust BoT-SORT tracker parameters:

track_high_thresh: 0.6
appearance_thresh: 0.3
with_reid: true
proximity_thresh: 0.6

🧠 Future Improvements

Fine-tuned ReID model for jersey consistency under occlusion

Automatic event recognition (goals, passes, tackles)

Multi-camera synchronization

Interactive web dashboard for analytics visualization

🏆 Acknowledgements

Built using the following technologies:

Ultralytics YOLOv11

BoT-SORT

TorchReID

SigLIP (Google Research)

OpenCV

Matplotlib

NumPy

🧾 Author

Youssef Esmael
📍 Egypt
📧 ismmailmuhamed@gmail.com

📜 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it with attribution.
