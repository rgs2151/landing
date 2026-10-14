"""Export the existing tutorial's processed recording, without regenerating it."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path, help="Existing mice_lfp.npz with xs, Fs and tau")
parser.add_argument("--output", type=Path, default=Path("public/assets/data/mouse-lfp.json"))
args = parser.parse_args()

with np.load(args.source, allow_pickle=False) as cached:
    samples = np.asarray(cached["xs"], dtype=np.float64)
    sample_rate = float(cached["Fs"])
    lag = int(cached["tau"])

if samples.ndim != 1 or not np.isfinite(samples).all():
    raise ValueError("Expected a finite, one-dimensional recording")
if len(samples) != 2500 or sample_rate != 500 or lag != 10:
    raise ValueError("This export expects the tutorial's five-second, 500 Hz, tau=10 cache")

payload = {
    "kind": "recorded-mouse-lfp",
    "source": "eke003_spont_250818_175844.npy",
    "channel": 1,
    "sourceStartSeconds": 0,
    "durationSeconds": len(samples) / sample_rate,
    "sampleRate": sample_rate,
    "lagSamples": lag,
    "notebook": "docs/tutorials/mouse_lfp.ipynb",
    "referenceCommit": "9eafbedcfbddb25f0d6102db6e75ddfa6eb31c79",
    "processing": "Tutorial xs: 50 Hz notch Q=30, 1-20 Hz bandpass, decimate 20000 to 500 Hz, 1-80 Hz envelope normalization with 3 Hz envelope lowpass",
    "samplesFloat64Sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
    "samples": samples.tolist(),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
print(f"Exported {len(samples)} unchanged samples; {args.output.stat().st_size} bytes")
