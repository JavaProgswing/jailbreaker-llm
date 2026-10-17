"""Download and normalize the official JailbreakBench JBB-Behaviors dataset."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests


DATASET = 'JailbreakBench/JBB-Behaviors'
CONFIG = 'behaviors'
SPLITS = ('harmful', 'benign')
VIEWER_BASE = 'https://datasets-server.huggingface.co'
DATASET_CARD = 'https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors'
EXPECTED_FIELDS = {'Index', 'Goal', 'Target', 'Behavior', 'Category', 'Source'}


def _fetch_split(split: str, timeout_s: float = 60.0) -> list[dict]:
    query = urlencode({
        'dataset': DATASET,
        'config': CONFIG,
        'split': split,
        'offset': 0,
        'length': 100,
    })
    response = requests.get(f'{VIEWER_BASE}/rows?{query}', timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    total = payload.get('num_rows_total', len(payload.get('rows', [])))
    rows = payload.get('rows', [])
    if total != 100 or len(rows) != total:
        raise RuntimeError(f'expected 100 {split} rows from {DATASET}, received {len(rows)}/{total}')
    normalized = []
    for wrapped in rows:
        row = wrapped['row']
        missing = EXPECTED_FIELDS.difference(row)
        if missing:
            raise RuntimeError(f'{split} row is missing required fields: {sorted(missing)}')
        normalized.append({
            'dataset': DATASET,
            'config': CONFIG,
            'split': split,
            'index': int(row['Index']),
            'goal': row['Goal'],
            'target': row['Target'],
            'behavior': row['Behavior'],
            'category': row['Category'],
            'source': row['Source'],
        })
    return normalized


def _write_jsonl(path: str, rows: list[dict]):
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(out_dir: str, smoke_count: int = 2):
    if smoke_count < 1 or smoke_count > 100:
        raise ValueError('smoke_count must be between 1 and 100')
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    counts = {}
    for split in SPLITS:
        rows = _fetch_split(split)
        full_path = os.path.join(out_dir, f'{split}.jsonl')
        smoke_path = os.path.join(out_dir, f'{split}_smoke.jsonl')
        seeds_path = os.path.join(out_dir, f'{split}_seeds.txt')
        _write_jsonl(full_path, rows)
        _write_jsonl(smoke_path, rows[:smoke_count])
        with open(seeds_path, 'w', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                handle.write(row['goal'].replace('\r', ' ').replace('\n', ' ').strip() + '\n')
        counts[split] = len(rows)
        files[os.path.basename(full_path)] = _sha256(full_path)
        files[os.path.basename(smoke_path)] = _sha256(smoke_path)
        files[os.path.basename(seeds_path)] = _sha256(seeds_path)
        print(f'wrote {len(rows)} {split} behaviors to {full_path}')
    manifest = {
        'dataset': DATASET,
        'config': CONFIG,
        'source': DATASET_CARD,
        'license': 'MIT',
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'counts': counts,
        'smoke_count': smoke_count,
        'sha256': files,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    print(f'wrote reproducibility manifest to {out_dir}/manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='data/jailbreakbench')
    parser.add_argument('--smoke_count', type=int, default=2)
    args = parser.parse_args()
    main(args.out_dir, args.smoke_count)
