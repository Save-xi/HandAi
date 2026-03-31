from pathlib import Path
import sys

# Ensure imports work whether pytest is run from this directory or repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
