import json
from collections import Counter


def build_report(transcript_path: str, campaign_id: str) -> dict:
    rows = [json.loads(l) for l in open(transcript_path)]
    if not rows:
        print('no rows in transcript, nothing to report')
        return {}
    conversations = {}
    for row in rows:
        conversations[row['seed_prompt']] = row
    attempts = list(conversations.values())
    label_counts = Counter((r['judge_label'] for r in attempts))
    total = len(attempts)
    success_rate = label_counts.get('full_violation', 0) / total
    partial_rate = label_counts.get('partial_compliance', 0) / total
    review_flagged = sum((1 for r in rows if r.get('eval_needs_human_review')))
    best_turn_by_seed = {}
    for row in rows:
        score = row.get('eval_ensemble', row['judge_scores']).get('full_violation', 0)
        current = best_turn_by_seed.get(row['seed_prompt'])
        if current is None or score > current[0]:
            best_turn_by_seed[row['seed_prompt']] = (score, row)
    worst_examples = [item[1] for item in sorted(best_turn_by_seed.values(), key=lambda item: item[0], reverse=True)[:5]]
    report = {
        'campaign_id': campaign_id,
        'target_model': rows[0].get('target_model', 'unknown'),
        'target_mode': rows[0].get('target_mode', 'unknown'),
        'total_attempts': total,
        'total_turns': len(rows),
        'full_violation_rate': round(success_rate, 3),
        'partial_compliance_rate': round(partial_rate, 3),
        'refusal_rate': round(label_counts.get('refusal', 0) / total, 3),
        'flagged_for_human_review': review_flagged,
        'single_judge_fallback': any((r.get('eval_single_judge_fallback') for r in rows)),
        'top_reproducible_examples': [{'attacker_prompt': r['attacker_prompt'], 'target_response': r['target_response'][:300], 'technique': r.get('technique', 'unknown'), 'ensemble_full_violation_score': round(r.get('eval_ensemble', r['judge_scores']).get('full_violation', 0), 3), 'needs_human_review': r.get('eval_needs_human_review', False)} for r in worst_examples],
    }
    benchmark_attempts = [row for row in attempts if isinstance(row.get('benchmark'), dict)]
    if benchmark_attempts:
        report['benchmark'] = {
            'datasets': sorted({row['benchmark'].get('dataset', 'unknown') for row in benchmark_attempts}),
            'splits': sorted({row['benchmark'].get('split', 'unknown') for row in benchmark_attempts}),
            'by_category': {},
        }
        categories = sorted({row['benchmark'].get('category', 'unknown') for row in benchmark_attempts})
        for category in categories:
            category_rows = [row for row in benchmark_attempts if row['benchmark'].get('category', 'unknown') == category]
            category_counts = Counter(row['judge_label'] for row in category_rows)
            count = len(category_rows)
            report['benchmark']['by_category'][category] = {
                'attempts': count,
                'full_violation_rate': round(category_counts.get('full_violation', 0) / count, 3),
                'partial_compliance_rate': round(category_counts.get('partial_compliance', 0) / count, 3),
                'refusal_rate': round(category_counts.get('refusal', 0) / count, 3),
            }
    out_path = f'data/transcripts/{campaign_id}_report.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'report written to {out_path}')
    print(f'full-violation rate: {success_rate:.1%}  |  partial: {partial_rate:.1%}  |  flagged for review: {review_flagged}')
    return report
