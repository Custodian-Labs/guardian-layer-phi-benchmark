"""Plot mouth-opening / blink / yaw over time: original vs each output (needs matplotlib)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
series = {}
for arg in sys.argv[2:]:               # label=path/to/X.series.json
    lab, p = arg.split("=", 1)
    d = json.loads(Path(p).read_text())
    series.setdefault("original", d["original"])
    series[lab] = d[lab]
tracks = sorted(series["original"])
fig, axes = plt.subplots(3, len(tracks), figsize=(7 * len(tracks), 8), squeeze=False)
cols = {"original": "#222222", "static": "#d08c2b", "reenact": "#2b6cd0"}
for j, tid in enumerate(tracks):
    for i, key in enumerate(("mouth_open", "eyeBlink", "yaw")):
        ax = axes[i, j]
        for lab, s in series.items():
            pts = [(f, m[key]) for f, m in s.get(tid, []) if m]
            if pts:
                ax.plot([p[0] / 25 for p in pts], [p[1] for p in pts], label=lab,
                        color=cols.get(lab), lw=1.4 if lab != "original" else 2.0)
        ax.set_title(f"{tid}: {key}"); ax.set_xlabel("s")
        if i == 0 and j == 0:
            ax.legend()
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote", out)
