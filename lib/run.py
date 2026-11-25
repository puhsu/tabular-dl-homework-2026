"""Run one homework experiment, or all six in order."""

import os
import subprocess
from pathlib import Path

import cyclopts
import torch

import train_kumo
import train_tabpack
from lib import experiment

DATASETS = {"california": "rmse", "otto": "cross-entropy"}
MODELS = ("tabpack-cosine", "tabpack-ple", "kumo-medium")


def main(exp: Path | None = None, *, colab: bool = False, force: bool = False):
    paths = [
        Path("experiments") / dataset / model
        for dataset in DATASETS
        for model in MODELS
    ]

    if colab:
        pending = [path for path in paths if force or not experiment.is_completed(path)]
        if pending:
            session = os.environ.get("COLAB_SESSION", "tabular-hw")
            subprocess.run(["sh", "lib/colab.sh", session, *pending], check=True)
        else:
            print("Already completed. Use --force to rerun.")
        return

    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path}", flush=True)
        config = experiment.load_config(path)
        if experiment.start(path, force=force):
            if path.name == "kumo-medium":
                report = train_kumo.evaluate(config)
            else:
                report = train_tabpack.train(config)

            device = torch.device(config["device"])
            report["device"] = (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device)
            )
            experiment.finish(path, report)
        else:
            report = experiment.load_report(path)

        metric = DATASETS[path.parent.name]
        val = report["metrics"]["val"][metric]
        test = report["metrics"]["test"][metric]
        print(
            f"{metric}: val={val:.5f}, test={test:.5f}, time={report['time']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    cyclopts.run(main)
