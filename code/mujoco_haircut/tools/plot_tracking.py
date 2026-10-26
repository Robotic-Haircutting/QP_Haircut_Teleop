from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from haircut_sim.tracking_report import make_tracking_report
from haircut_sim.report import latest_csv, default_log_dir
if __name__ == "__main__":
    make_tracking_report(Path(sys.argv[1]) if len(sys.argv)>1 else latest_csv(default_log_dir()))
