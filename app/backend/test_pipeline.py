import sys
import time

sys.path.insert(0, ".")
from ai_pipeline import run_phase1, run_phase2

VIDEO_PATH = "uploads/20260531_131548_0001.mp4"

print("=== Phase 1 ===")
t0 = time.time()
segments = run_phase1(VIDEO_PATH)
print(f"Time: {time.time() - t0:.1f}s")
print(f"Segments found: {len(segments)}")
for s in segments:
    print(
        f"  {s['start_time']:.1f}s - {s['end_time']:.1f}s | "
        f"score: {s['anomaly_score']:.3f}"
    )

if segments:
    print("\n=== Phase 2 (segment 0) ===")
    t0 = time.time()
    result = run_phase2(VIDEO_PATH, segments[0])
    print(f"Time: {time.time() - t0:.1f}s")
    print(f"Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence_score']:.3f}")
