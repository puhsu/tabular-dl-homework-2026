"""Small experiment directories: config.json in, report.json out."""

import json
from pathlib import Path


def load_config(exp):
    return json.loads((Path(exp) / "config.json").read_text())


def load_report(exp):
    return json.loads((Path(exp) / "report.json").read_text())


def dump_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def create(exp, config):
    exp = Path(exp)
    exp.mkdir(parents=True, exist_ok=True)
    if (exp / "config.json").exists():
        if load_config(exp) != config:
            raise ValueError(f"Existing configuration differs: {exp}")
    else:
        dump_json(exp / "config.json", config)
    return exp


def is_completed(exp):
    return (Path(exp) / "DONE").exists()


def start(exp, *, force=False):
    exp = Path(exp)
    if is_completed(exp) and not force:
        print(f"Already completed: {exp}")
        return False
    if (exp / "RUNNING").exists() and not force:
        raise RuntimeError(f"Interrupted experiment: {exp}; use --force")
    (exp / "DONE").unlink(missing_ok=True)
    (exp / "RUNNING").touch()
    return True


def finish(exp, report):
    exp = Path(exp)
    dump_json(exp / "report.json", report)
    (exp / "RUNNING").unlink(missing_ok=True)
    (exp / "DONE").touch()
