#!/bin/sh
# Arguments: existing session name, then the experiment directories to run.
set -eu
session=$1
shift
colab() {
    command=$1
    shift
    uvx --from google-colab-cli colab "$command" -s "$session" "$@"
}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
job=/content/tabular-$(basename "$work")
COPYFILE_DISABLE=1 tar -czf "$work/code.tar.gz" *.py lib/*.py pyproject.toml uv.lock \
    experiments/*/*/config.json data/fetch.sh data/SHA256SUMS weights/fetch.sh weights/SHA256SUMS
colab upload "$work/code.tar.gz" "$job.tar.gz"
colab ssh -- "mkdir '$job' && cd '$job' && tar -xzf '$job.tar.gz' && sh data/fetch.sh"
case " $* " in
    *'/kumo-medium '*) colab ssh -- "cd '$job' && sh weights/fetch.sh" ;;
esac

for exp do
    echo "Running $exp on $session"
    colab ssh -- "cd '$job' && uv run --frozen python -u -m lib.run '$exp'"
    colab download "$job/$exp/report.json" "$work/report.json"
    mv "$work/report.json" "$exp/report.json"
    touch "$exp/DONE"
done
