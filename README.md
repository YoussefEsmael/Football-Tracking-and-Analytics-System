# ⚽ Football Tracking and Analytics System

A complete end-to-end system for **football (soccer) video analytics**, integrating **object detection, player tracking, team classification, and advanced match analytics** — including positional heatmaps and performance visualizations.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YoussefEsmael/Football-Tracking-and-Analytics-System.git
cd Football-Tracking-and-Analytics-System

# (Optional) Create a virtual environment
conda create -n football_tracker python=3.10
conda activate football_tracker

# Install dependencies
pip install -r requirements.txt

## ▶️ Usage
#python src/main_enhanced.py --video "path/to/match.mp4"
Example
python src/main_enhanced.py \
  --video "input/match.mp4" \
  --out "output/processed_match.mp4" \
  --model "models/best (4).pt" \
  --tracker-config "configs/botsort.yaml" \
  --no-siglip
