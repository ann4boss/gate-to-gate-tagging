"""
output/formatter.py
-------------------
Converts raw passage events into the JSON schema used by the
manually-tagged dataset.
"""

from typing import List, Optional


def build_output(run_id: str, fps: float, passages: List[dict]) -> dict:
    """
    Build the final JSON-serialisable result dict.

    Parameters
    ----------
    run_id   : identifier for this run (e.g. "ATHLETE_001_RUN1")
    fps      : video frame rate
    passages : sorted list of passage events from logic/events.py

    Returns
    -------
    dict matching the manually-tagged schema:
        {run_id, fps, gates: [{gate_number, gate_label, position_ms,
                               position_s, frame, duration_ms, duration_s}]}
    """
    gates = []

    for i, ev in enumerate(passages):
        frame  = ev["frame"]
        pos_ms = _frame_to_ms(frame, fps)
        pos_s  = round(pos_ms / 1000, 3)

        # duration = gap to next gate passage (None for last gate)
        if i + 1 < len(passages):
            next_ms = _frame_to_ms(passages[i + 1]["frame"], fps)
            dur_ms  = round(next_ms - pos_ms, 0)
            dur_s   = round(dur_ms / 1000, 3)
        else:
            dur_ms = None
            dur_s  = None

        gates.append({
            "gate_number": i + 1,
            "gate_label":  f"Gate {i + 1}",
            "position_ms": pos_ms,
            "position_s":  pos_s,
            "frame":       frame,
            "duration_ms": dur_ms,
            "duration_s":  dur_s,
            # extra diagnostic fields (not in manual dataset — remove if needed)
            "_gate_tracker_id": ev.get("gate_id"),
            "_min_dist_px":     round(ev.get("min_dist_px", 0.0), 1),
        })

    return {
        "run_id": run_id,
        "fps":    fps,
        "gates":  gates,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _frame_to_ms(frame: int, fps: float) -> float:
    return round(frame / fps * 1000, 0)