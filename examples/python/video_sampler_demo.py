from __future__ import annotations

import argparse
import json

from machboost import TemporalVideoSampler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare uniform and temporal-change frame selection for one video."
    )
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--change-threshold", type=float, default=0.08)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()

    sampler = TemporalVideoSampler()
    uniform = sampler.sample_uniform(
        args.video,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    temporal = sampler.sample(
        args.video,
        fps=args.fps,
        change_threshold=args.change_threshold,
        max_frames=args.max_frames,
    )
    print(
        json.dumps(
            {
                "uniform": uniform.to_dict(),
                "temporal": temporal.to_dict(),
                "frame_reduction_rate": (
                    0.0
                    if uniform.selected_frames == 0
                    else 1.0 - temporal.selected_frames / uniform.selected_frames
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
