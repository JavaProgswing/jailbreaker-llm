import json

from src.eval.redact import redact_turn
from src.eval.report import build_report
from src.eval.run_campaign import load_seed_cases


def test_load_seed_cases_preserves_jailbreakbench_metadata_without_target(tmp_path):
    path = tmp_path / 'jbb.jsonl'
    path.write_text(json.dumps({
        'dataset': 'JailbreakBench/JBB-Behaviors',
        'split': 'harmful',
        'index': 7,
        'goal': 'benchmark goal',
        'target': 'reference target that must not enter transcript metadata',
        'behavior': 'behavior-id',
        'category': 'category-a',
        'source': 'Original',
    }) + '\n', encoding='utf-8')

    case = load_seed_cases(str(path))[0]

    assert case.prompt == 'benchmark goal'
    assert case.benchmark['dataset'] == 'JailbreakBench/JBB-Behaviors'
    assert case.benchmark['index'] == 7
    assert 'target' not in case.benchmark


def test_redaction_also_covers_seed_prompt():
    row = redact_turn({'seed_prompt': 'contact person@example.com', 'attacker_prompt': 'safe', 'target_response': 'safe'})
    assert row['seed_prompt'] == 'contact [REDACTED:EMAIL]'


def test_report_counts_conversations_and_groups_benchmark_categories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    transcript_dir = tmp_path / 'data' / 'transcripts'
    transcript_dir.mkdir(parents=True)
    rows = [
        {'seed_prompt': 'a', 'judge_label': 'refusal', 'judge_scores': {'full_violation': 0.1}, 'eval_ensemble': {'full_violation': 0.1}, 'attacker_prompt': 'a1', 'target_response': 'r1', 'benchmark': {'dataset': 'JBB', 'split': 'harmful', 'category': 'cat-a'}},
        {'seed_prompt': 'a', 'judge_label': 'refusal', 'judge_scores': {'full_violation': 0.2}, 'eval_ensemble': {'full_violation': 0.2}, 'attacker_prompt': 'a2', 'target_response': 'r2', 'benchmark': {'dataset': 'JBB', 'split': 'harmful', 'category': 'cat-a'}},
        {'seed_prompt': 'b', 'judge_label': 'full_violation', 'judge_scores': {'full_violation': 0.8}, 'eval_ensemble': {'full_violation': 0.8}, 'attacker_prompt': 'b1', 'target_response': 'r3', 'benchmark': {'dataset': 'JBB', 'split': 'harmful', 'category': 'cat-b'}},
    ]
    transcript = transcript_dir / 'campaign.jsonl'
    transcript.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    report = build_report(str(transcript), 'campaign')

    assert report['total_attempts'] == 2
    assert report['total_turns'] == 3
    assert report['full_violation_rate'] == 0.5
    assert report['benchmark']['by_category']['cat-a']['refusal_rate'] == 1.0
    assert report['benchmark']['by_category']['cat-b']['full_violation_rate'] == 1.0
