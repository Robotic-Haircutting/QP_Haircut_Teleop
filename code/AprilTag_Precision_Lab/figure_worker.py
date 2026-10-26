"""Generate experiment figures and the session report in a separate process.

Figure generation (matplotlib) is CPU-heavy. Running it in a thread of the control service starved the
python-can receive thread (one GIL), so CAN feedback looked stale for >0.5 s and the safety checks put the
arm into emergency stop right after Save. A separate process cannot block the control process.

Usage (called by robot_bridge.finish_log): python figure_worker.py <args.json>
Prints one JSON line: {"plots": ..., "report": ..., "error": ...}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    args = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = {"plots": None, "report": None, "error": None}
    try:
        from experiment_figures import make_figures
        out["plots"] = make_figures(args["path"], args["mode"])
        from session_report import export_session
        out["report"] = export_session(args["path"], args["mode"], args["started"], alignment=args.get("alignment"))
    except Exception as e:  # noqa: BLE001 - reported back to the service
        out["error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
