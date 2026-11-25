"""Run the six experiments once, reuse reports, and rerun one with --force."""

from lib import experiment, run


def test_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []

    def report(config):
        calls.append(config["model"])
        metric = run.DATASETS[config["dataset"]]
        return {
            "time": 1,
            "metrics": {split: {metric: 0.5} for split in ("val", "test")},
        }

    monkeypatch.setattr(run.train_tabpack, "train", report)
    monkeypatch.setattr(run.train_kumo, "evaluate", report)
    for dataset in run.DATASETS:
        for model in run.MODELS:
            experiment.create(
                tmp_path / "experiments" / dataset / model,
                {"dataset": dataset, "model": model, "device": "cpu"},
            )
    run.main()
    run.main()
    assert calls == list(run.MODELS) * 2
    run.main(tmp_path / "experiments/otto/kumo-medium", force=True)
    assert calls == [*run.MODELS, *run.MODELS, "kumo-medium"]
