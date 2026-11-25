#!/bin/sh
set -e
cd "$(dirname "$0")"

url=https://huggingface.co/datasets/puhsu/tabular-dl-homework-2026/resolve/main
while read -r hash file; do
    if [ ! -f "$file" ]; then
        curl --fail --location --create-dirs --output "$file" "$url/$file"
    fi
done < SHA256SUMS

shasum -a 256 -c SHA256SUMS
