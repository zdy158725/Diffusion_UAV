import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_file_loader
except ImportError as exc:
    raise SystemExit(
        "tensorboard is required to backfill analysis from event files. "
        "Use an environment that has the tensorboard package installed."
    ) from exc


NAMESPACE_MAP = {
    'train': 'train',
    'eval': 'eval',
    'eval/loss': 'eval_loss',
}

METRICS_BY_NAMESPACE = {
    'train': {'ade', 'fde', 'kde', 'obs_viols'},
    'eval': {'ade', 'fde', 'kde', 'obs_viols'},
    'eval/loss': {'nll_q_is', 'nll_p', 'nll_exact', 'nll_sampled'},
}

LOWER_IS_BETTER = {
    'ade',
    'fde',
    'kde',
    'obs_viols',
    'nll_q_is',
    'nll_p',
    'nll_exact',
    'nll_sampled',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-dir',
        type=Path,
        required=True,
        help='Path to a training run directory that contains tensorboard/ and analysis/.',
    )
    return parser.parse_args()


def ensure_nested(mapping, *keys):
    current = mapping
    for key in keys:
        if key not in current:
            current[key] = {}
        current = current[key]
    return current


def parse_histogram_value(histogram_proto):
    return {
        'count': int(round(histogram_proto.num)),
        'min': float(histogram_proto.min),
        'max': float(histogram_proto.max),
        'sum': float(histogram_proto.sum),
        'sum_squares': float(histogram_proto.sum_squares),
    }


def merge_metric_summary(histogram_stats, scalar_stats):
    summary = {}
    if histogram_stats is not None:
        count = histogram_stats['count']
        hist_mean = histogram_stats['sum'] / count if count else None
        variance = None
        if count:
            variance = histogram_stats['sum_squares'] / count - hist_mean * hist_mean
            variance = max(variance, 0.0)
        summary.update(
            {
                'count': count,
                'min': histogram_stats['min'],
                'max': histogram_stats['max'],
            }
        )
        if hist_mean is not None:
            summary['mean'] = hist_mean
        if variance is not None:
            summary['std'] = math.sqrt(variance)

    if scalar_stats:
        if 'mean' in scalar_stats:
            summary['mean'] = float(scalar_stats['mean'])
        if 'median' in scalar_stats:
            summary['median'] = float(scalar_stats['median'])

    ordered_keys = ['count', 'mean', 'median', 'std', 'min', 'max']
    return {key: summary[key] for key in ordered_keys if key in summary}


def parse_summary_tag(tag):
    parts = tag.split('/')
    if len(parts) < 3:
        return None

    node_name = parts[0]
    namespace = '/'.join(parts[1:-1])
    metric_name = parts[-1]
    if namespace not in NAMESPACE_MAP:
        return None

    if metric_name.endswith('_mean'):
        base_metric_name = metric_name[:-5]
        if base_metric_name not in METRICS_BY_NAMESPACE[namespace]:
            return None
        return NAMESPACE_MAP[namespace], node_name, base_metric_name, 'mean'
    if metric_name.endswith('_median'):
        base_metric_name = metric_name[:-7]
        if base_metric_name not in METRICS_BY_NAMESPACE[namespace]:
            return None
        return NAMESPACE_MAP[namespace], node_name, base_metric_name, 'median'
    if metric_name not in METRICS_BY_NAMESPACE[namespace]:
        return None
    return NAMESPACE_MAP[namespace], node_name, metric_name, 'histogram'


def parse_event_files(event_files):
    histogram_data = defaultdict(dict)
    scalar_data = defaultdict(dict)
    auxiliary_scalars = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for event_file in event_files:
        loader = event_file_loader.LegacyEventFileLoader(str(event_file))
        for event in loader.Load():
            if not event.HasField('summary'):
                continue

            step = int(event.step)
            for value in event.summary.value:
                parsed = parse_summary_tag(value.tag)
                if parsed is not None:
                    namespace, node_name, metric_name, value_kind = parsed
                    key = (namespace, node_name, metric_name)
                    if value_kind == 'histogram' and value.HasField('histo'):
                        histogram_data[step][key] = parse_histogram_value(value.histo)
                    elif value_kind in ('mean', 'median') and value.HasField('simple_value'):
                        scalar_entry = scalar_data[step].setdefault(key, {})
                        scalar_entry[value_kind] = float(value.simple_value)
                    continue

                if value.HasField('simple_value'):
                    if value.tag == 'train/learning_rate':
                        auxiliary_scalars[step]['learning_rate']['global'].append(float(value.simple_value))
                    elif value.tag.endswith('/train/loss'):
                        node_name = value.tag.split('/', 1)[0]
                        auxiliary_scalars[step]['train_loss'][node_name].append(float(value.simple_value))
                    elif value.tag.endswith('/log_likelihood_eval'):
                        node_name = value.tag.split('/', 1)[0]
                        auxiliary_scalars[step]['log_likelihood_eval'][node_name].append(float(value.simple_value))

    return histogram_data, scalar_data, auxiliary_scalars


def build_snapshots(histogram_data, scalar_data):
    snapshots = []
    for step in sorted(set(histogram_data) | set(scalar_data)):
        summaries = {}
        step_keys = set(histogram_data.get(step, {})) | set(scalar_data.get(step, {}))
        for namespace, node_name, metric_name in sorted(step_keys):
            metric_summary = merge_metric_summary(
                histogram_data.get(step, {}).get((namespace, node_name, metric_name)),
                scalar_data.get(step, {}).get((namespace, node_name, metric_name)),
            )
            if not metric_summary:
                continue
            namespace_summary = ensure_nested(summaries, namespace)
            node_summary = ensure_nested(namespace_summary, node_name)
            node_summary[metric_name] = metric_summary

        if not summaries:
            continue

        snapshots.append(
            {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'iteration': step + 1,
                'source_step': step,
                'source': 'tensorboard_backfill',
                'summaries': summaries,
            }
        )
    return snapshots


def series_for_metric(snapshots, namespace, metric_name):
    series = defaultdict(list)
    for snapshot in snapshots:
        namespace_summary = snapshot['summaries'].get(namespace, {})
        for node_name, metrics in namespace_summary.items():
            metric_summary = metrics.get(metric_name)
            if metric_summary is None or 'mean' not in metric_summary:
                continue
            series[node_name].append((snapshot['iteration'], float(metric_summary['mean'])))
    return dict(series)


def average_series_by_iteration(series_by_node):
    per_iteration = defaultdict(list)
    for series in series_by_node.values():
        for iteration, value in series:
            per_iteration[iteration].append(value)
    return sorted((iteration, sum(values) / len(values)) for iteration, values in per_iteration.items() if values)


def summarize_series(series, lower_is_better=True):
    if not series:
        return None

    first_iteration, first_value = series[0]
    final_iteration, final_value = series[-1]
    comparator = min if lower_is_better else max
    best_iteration, best_value = comparator(series, key=lambda item: item[1])
    improvement_abs = first_value - best_value if lower_is_better else best_value - first_value
    improvement_pct = 0.0 if first_value == 0 else 100.0 * improvement_abs / abs(first_value)
    final_change_abs = first_value - final_value if lower_is_better else final_value - first_value
    final_change_pct = 0.0 if first_value == 0 else 100.0 * final_change_abs / abs(first_value)

    return {
        'first': {'iteration': first_iteration, 'value': first_value},
        'best': {'iteration': best_iteration, 'value': best_value},
        'final': {'iteration': final_iteration, 'value': final_value},
        'improvement_to_best_abs': improvement_abs,
        'improvement_to_best_pct': improvement_pct,
        'change_to_final_abs': final_change_abs,
        'change_to_final_pct': final_change_pct,
    }


def mean_or_none(values):
    return None if not values else sum(values) / len(values)


def build_auxiliary_overview(auxiliary_scalars):
    overview = {}
    for aux_name, lower_is_better in (
        ('train_loss', True),
        ('log_likelihood_eval', False),
        ('learning_rate', True),
    ):
        per_node_series = defaultdict(list)
        for step in sorted(auxiliary_scalars):
            for node_name, values in auxiliary_scalars[step].get(aux_name, {}).items():
                value = mean_or_none(values)
                if value is None:
                    continue
                per_node_series[node_name].append((step + 1, value))

        if not per_node_series:
            continue

        aux_summary = {}
        for node_name, series in sorted(per_node_series.items()):
            aux_summary[node_name] = summarize_series(series, lower_is_better=lower_is_better)
        overview[aux_name] = aux_summary

    return overview


def build_overview(snapshots, auxiliary_scalars):
    overview = {
        'num_eval_points': len(snapshots),
        'iteration_range': [snapshots[0]['iteration'], snapshots[-1]['iteration']] if snapshots else [],
        'aggregate': {},
        'per_node': {},
        'auxiliary': build_auxiliary_overview(auxiliary_scalars),
        'warnings': [],
    }

    per_node_metrics = defaultdict(dict)
    for namespace, metric_names in (
        ('eval', ('ade', 'fde', 'kde', 'obs_viols')),
        ('train', ('ade', 'fde')),
        ('eval_loss', ('nll_q_is', 'nll_p', 'nll_exact', 'nll_sampled')),
    ):
        for metric_name in metric_names:
            series_by_node = series_for_metric(snapshots, namespace, metric_name)
            if namespace == 'eval':
                aggregate_series = average_series_by_iteration(series_by_node)
                if aggregate_series:
                    aggregate_namespace = overview['aggregate'].setdefault(namespace, {})
                    aggregate_namespace[metric_name] = summarize_series(
                        aggregate_series,
                        lower_is_better=metric_name in LOWER_IS_BETTER,
                    )

            for node_name, series in series_by_node.items():
                node_summary = overview['per_node'].setdefault(node_name, {})
                namespace_summary = node_summary.setdefault(namespace, {})
                namespace_summary[metric_name] = summarize_series(
                    series,
                    lower_is_better=metric_name in LOWER_IS_BETTER,
                )
                per_node_metrics[node_name][(namespace, metric_name)] = series

    for node_name, node_summary in sorted(overview['per_node'].items()):
        final_gap = {}
        for metric_name in ('ade', 'fde'):
            train_series = per_node_metrics.get(node_name, {}).get(('train', metric_name))
            eval_series = per_node_metrics.get(node_name, {}).get(('eval', metric_name))
            if not train_series or not eval_series:
                continue
            train_final = train_series[-1][1]
            eval_final = eval_series[-1][1]
            final_gap[metric_name] = {
                'train_final': train_final,
                'eval_final': eval_final,
                'gap_abs': eval_final - train_final,
                'gap_pct_of_eval': 0.0 if eval_final == 0 else 100.0 * (eval_final - train_final) / abs(eval_final),
            }
        if final_gap:
            node_summary['final_train_eval_gap'] = final_gap

        nll_summary = node_summary.get('eval_loss', {}).get('nll_q_is')
        if nll_summary is not None:
            if nll_summary['final']['value'] > 0 and nll_summary['best']['value'] < 0:
                overview['warnings'].append(
                    f"{node_name} 的 eval_loss/nll_q_is 从 {nll_summary['best']['value']:.3f} "
                    f"恶化到 {nll_summary['final']['value']:.3f}。"
                )

    aggregate_eval = overview['aggregate'].get('eval', {})
    ade_summary = aggregate_eval.get('ade')
    fde_summary = aggregate_eval.get('fde')
    if ade_summary is not None and ade_summary['improvement_to_best_pct'] < 2.0:
        overview['warnings'].append(
            f"平均 eval ADE 最佳改善仅 {ade_summary['improvement_to_best_pct']:.2f}%，整体收益偏有限。"
        )
    if fde_summary is not None and fde_summary['improvement_to_best_pct'] < 2.0:
        overview['warnings'].append(
            f"平均 eval FDE 最佳改善仅 {fde_summary['improvement_to_best_pct']:.2f}%，整体收益偏有限。"
        )

    return overview


def render_overview_markdown(run_dir, overview):
    lines = [
        '# Experiment Analysis',
        '',
        f'- Run directory: `{run_dir}`',
        f"- Eval points: {overview.get('num_eval_points', 0)}",
    ]

    iteration_range = overview.get('iteration_range', [])
    if iteration_range:
        lines.append(f"- Iteration range: {iteration_range[0]} -> {iteration_range[1]}")

    aggregate_eval = overview.get('aggregate', {}).get('eval', {})
    for metric_name in ('ade', 'fde'):
        summary = aggregate_eval.get(metric_name)
        if summary is None:
            continue
        lines.extend(
            [
                '',
                f'## Average Eval {metric_name.upper()}',
                '',
                f"- First: iter {summary['first']['iteration']} -> {summary['first']['value']:.3f}",
                f"- Best: iter {summary['best']['iteration']} -> {summary['best']['value']:.3f}",
                f"- Final: iter {summary['final']['iteration']} -> {summary['final']['value']:.3f}",
                f"- Best improvement: {summary['improvement_to_best_abs']:.3f} ({summary['improvement_to_best_pct']:.2f}%)",
            ]
        )

    for node_name, node_summary in sorted(overview.get('per_node', {}).items()):
        lines.extend(['', f'## {node_name}', ''])
        for namespace, metric_names in (
            ('eval', ('ade', 'fde')),
            ('eval_loss', ('nll_q_is', 'nll_p')),
        ):
            for metric_name in metric_names:
                summary = node_summary.get(namespace, {}).get(metric_name)
                if summary is None:
                    continue
                lines.append(
                    f"- {namespace}/{metric_name}: first {summary['first']['value']:.3f} "
                    f"(iter {summary['first']['iteration']}), best {summary['best']['value']:.3f} "
                    f"(iter {summary['best']['iteration']}), final {summary['final']['value']:.3f} "
                    f"(iter {summary['final']['iteration']})"
                )
        for metric_name, gap in sorted(node_summary.get('final_train_eval_gap', {}).items()):
            lines.append(
                f"- final train/eval gap {metric_name}: {gap['gap_abs']:.3f} "
                f"({gap['gap_pct_of_eval']:.3f}% of eval)"
            )

    warnings = overview.get('warnings', [])
    if warnings:
        lines.extend(['', '## Warnings', ''])
        for warning in warnings:
            lines.append(f'- {warning}')

    lines.append('')
    return '\n'.join(lines)


def write_analysis_outputs(run_dir, snapshots, overview):
    analysis_dir = run_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for stale_snapshot in analysis_dir.glob('metrics_iter_*.json'):
        stale_snapshot.unlink()

    history_path = analysis_dir / 'metrics_history.jsonl'
    with history_path.open('w', encoding='utf-8') as history_file:
        for snapshot in snapshots:
            per_iter_path = analysis_dir / f"metrics_iter_{snapshot['iteration']:06d}.json"
            with per_iter_path.open('w', encoding='utf-8') as per_iter_file:
                json.dump(snapshot, per_iter_file, indent=2, sort_keys=True)
            history_file.write(json.dumps(snapshot, sort_keys=True))
            history_file.write('\n')

    if snapshots:
        latest_path = analysis_dir / 'metrics_latest.json'
        with latest_path.open('w', encoding='utf-8') as latest_file:
            json.dump(snapshots[-1], latest_file, indent=2, sort_keys=True)

    overview_path = analysis_dir / 'overview.json'
    with overview_path.open('w', encoding='utf-8') as overview_file:
        json.dump(overview, overview_file, indent=2, sort_keys=True)

    report_path = analysis_dir / 'report.md'
    with report_path.open('w', encoding='utf-8') as report_file:
        report_file.write(render_overview_markdown(run_dir, overview))


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    tensorboard_dir = run_dir / 'tensorboard'
    if not tensorboard_dir.is_dir():
        raise SystemExit(f"tensorboard directory not found: {tensorboard_dir}")

    event_files = sorted(tensorboard_dir.glob('events.out.tfevents.*'))
    if not event_files:
        raise SystemExit(f"no TensorBoard event files found under: {tensorboard_dir}")

    histogram_data, scalar_data, auxiliary_scalars = parse_event_files(event_files)
    snapshots = build_snapshots(histogram_data, scalar_data)
    if not snapshots:
        raise SystemExit("no backfillable metrics were found in the TensorBoard event files")

    overview = build_overview(snapshots, auxiliary_scalars)
    write_analysis_outputs(run_dir, snapshots, overview)

    print(f"Backfilled {len(snapshots)} analysis snapshots into {run_dir / 'analysis'}")
    print(f"Latest iteration: {snapshots[-1]['iteration']}")


if __name__ == '__main__':
    main()
