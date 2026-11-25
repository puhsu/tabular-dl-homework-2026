#!/bin/sh
set -e
cd "$(dirname "$0")"

url=https://huggingface.co/nvidia/Kumo-Tabular/resolve/main/medium
for file in regressor.pt classifier.pt; do
    if [ ! -f "$file" ]; then
        curl --fail --location --output "$file" "$url/$file"
    fi
done

shasum -a 256 -c SHA256SUMS
