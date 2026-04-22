"""Running/stalled/stopped classification from progress.json + history."""

from __future__ import annotations

import statistics
import time


def classify(
    progress: dict | None,
    recent_gen_times: list[float],
) -> dict:
    """Classify an experiment's liveness.

    Uses the median of recent gen durations as the cadence baseline, then:
      RUNNING : age <= median * 1.5
      STALLED : median * 1.5 < age <= median * 4
      STOPPED : age >  median * 4

    Falls back to hardcoded windows (2 min / 10 min) when gen_time history
    is missing — common for brand-new experiments before the first gen
    completes.
    """
    if not progress:
        return {"state": "UNKNOWN", "age_s": None, "cadence_s": None, "reason": "no progress.json"}

    ts = progress.get("timestamp")
    if ts is None:
        return {"state": "UNKNOWN", "age_s": None, "cadence_s": None, "reason": "no timestamp"}

    age = max(0.0, time.time() - float(ts))

    # Prefer recent-history median (robust to outliers and init-time spikes).
    times = [t for t in recent_gen_times if t and t > 0]
    if times:
        cadence = statistics.median(times)
    else:
        # Fallback when no history yet — very generous window.
        cadence = 120.0

    running_cutoff = cadence * 1.5
    stopped_cutoff = cadence * 4.0

    if age <= running_cutoff:
        state = "RUNNING"
    elif age <= stopped_cutoff:
        state = "STALLED"
    else:
        state = "STOPPED"

    return {
        "state": state,
        "age_s": age,
        "cadence_s": cadence,
        "reason": None,
    }
