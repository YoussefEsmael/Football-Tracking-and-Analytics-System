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
▶️ Usage
Run the full analytics pipeline:

bash
Copy code
python src/main_enhanced.py --video "path/to/match.mp4"
Example:

bash
Copy code
python src/main_enhanced.py \
  --video "input/match.mp4" \
  --out "output/processed_match.mp4" \
  --model "models/best (4).pt" \
  --tracker-config "configs/botsort.yaml" \
  --no-siglip
yaml
Copy code

---
