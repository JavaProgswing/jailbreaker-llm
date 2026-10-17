# JailbreakBench data

Run `python data/import_jailbreakbench.py` from the repository root to fetch
the official `JailbreakBench/JBB-Behaviors` dataset through the Hugging Face
Dataset Viewer API.

Generated files:

- `harmful.jsonl` and `benign.jsonl`: all 100 behaviors in each split.
- `harmful_smoke.jsonl` and `benign_smoke.jsonl`: first two rows by default.
- `harmful_seeds.txt` and `benign_seeds.txt`: plain-text compatibility files.
- `manifest.json`: retrieval time, row counts, source, license, and SHA-256
  hashes.

The JSONL files preserve `index`, `goal`, `target`, `behavior`, `category`, and
`source`. Campaigns use `goal` as the seed and retain non-prompt benchmark
metadata in transcript rows.

Use JailbreakBench as a held-out evaluation set. Do not train the attacker or
reward judge on the same behavior rows and then report results on them.

Some dataset contents may be offensive. Store and share generated transcripts
according to your organization's authorized red-team data policy.

Dataset: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors

License: MIT

Citation: Chao et al., "JailbreakBench: An Open Robustness Benchmark for
Jailbreaking Large Language Models," NeurIPS Datasets and Benchmarks Track,
2024.
