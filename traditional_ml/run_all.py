import argparse

import subprocess

import sys


from . import config


def _run(module: str, extra: list[str]) -> None:

    cmd = [sys.executable, "-m", f"traditional_ml.{module}"] + extra

    print(f"\n$ {' '.join(cmd)}", flush=True)

    subprocess.run(cmd, check=True)


def main():

    p = argparse.ArgumentParser(description="Run the whole traditional-ML pipeline locally")

    p.add_argument("--datasets", nargs="+", default=config.DATASETS)

    p.add_argument("--models", nargs="+", default=config.MODEL_NAMES)

    p.add_argument("--sample", type=int, default=config.SAMPLE)

    p.add_argument("--val_docs", type=int, default=config.VAL_DOCS)

    p.add_argument("--hparam_subsample", type=int, default=None)

    p.add_argument("--no_bertscore", action="store_true")

    p.add_argument("--force", action="store_true")

    args = p.parse_args()


    sample = [] if args.sample is None else ["--sample", str(args.sample)]

    hps = [] if args.hparam_subsample is None else ["--hparam_subsample", str(args.hparam_subsample)]

    val = [] if args.val_docs is None else ["--val_docs", str(args.val_docs)]


    for dataset in args.datasets:

        _run("train", ["--dataset", dataset, "--models", *args.models]

             + val + sample + hps + (["--force"] if args.force else []))

    for name in args.models:

        _run("generate", ["--model", name, "--datasets", *args.datasets] + sample)

    _run("evaluate", ["--models", *args.models, "--datasets", *args.datasets]

         + (["--no_bertscore"] if args.no_bertscore else []))

    _run("aggregate", [])

    print("\nAll done.", flush=True)


if __name__ == "__main__":

    main()
