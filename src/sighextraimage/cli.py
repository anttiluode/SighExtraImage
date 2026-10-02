from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sighextraimage",
        description="SighExtraImage: physics-guided out-of-frame inference",
    )
    sub = parser.add_subparsers(dest="command")
    for name in ("synth", "extract", "invert", "gate0", "gate1"):
        sub.add_parser(name)
    gui = sub.add_parser("gui")
    gui.add_argument("--share", action="store_true")
    gui.add_argument("--port", type=int, default=None)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    bench.add_argument("--candidates", type=int, default=1200)
    bench.add_argument("--sigma", type=float, default=0.01)
    bench.add_argument("--residual-mode", choices=["l2", "affine", "affine_per_channel"], default="affine")
    bench.add_argument("--output", default="results/receipts/gates-0-1-v0.json")
    bench.add_argument("--tv-iters", type=int, default=250)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "gui":
        from .gui import launch_gui
        launch_gui(share=args.share, server_port=args.port)
        return 0
    if args.command == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(
            args.seeds,
            candidates=args.candidates,
            sigma=args.sigma,
            residual_mode=args.residual_mode,
            output=Path(args.output),
            tv_iters=args.tv_iters,
        )
        print(f"wrote {args.output}")
        return 0
    raise NotImplementedError(f"{args.command} is not implemented yet")
