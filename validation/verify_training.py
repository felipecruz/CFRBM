"""Verify F10/F11 evidence and compare final/best scores to the preceding run.

Usage: .venv/bin/python validation/verify_training.py f10 [implementation-commit]
The optional commit verifies every captured source file against Git. This
checks evidence integrity; it does not impose an accuracy target on a bug fix.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULT = 'initial_experiment_100k-u1xu1.json'


def read_run(feature):
    evidence = ROOT / 'validation' / feature
    metadata = json.loads((evidence / 'run.json').read_text())
    result = json.loads((evidence / RESULT).read_text())
    config = json.loads((ROOT / 'ubased.json').read_text())['experiments'][0]['configs'][0]
    assert metadata['exit_code'] == 0, 'Training did not finish'
    for name, digest in metadata['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert {key: value for key, value in result.items() if key != 'results'} == config
    rows = result['results']
    assert len(rows) == config['epochs'] == 100
    assert [row['iteration'] for row in rows] == list(range(100))
    logged = re.findall(
        r'^End iter (\d+) - k/lr: (\d+)/([\d.e+-]+) momentum: ([\d.e+-]+) '
        r'- MAE/RMSE: ([\d.e+-]+)/([\d.e+-]+)$',
        (evidence / 'training.log').read_text(), re.MULTILINE)
    assert len(logged) == len(rows)
    for row, log in zip(rows, logged):
        assert all(math.isfinite(row[key]) for key in ('mae', 'rmse'))
        assert 0 <= row['mae'] <= row['rmse'] <= 4
        assert row['k'] == 1 and row['lrate'] == .0005
        assert row['momentum'] == (.5 if row['iteration'] < 50 else .6)
        assert tuple(map(float, log)) == tuple(row[key] for key in
                                             ('iteration', 'k', 'lrate', 'momentum', 'mae', 'rmse'))
    return metadata, rows


def main():
    feature = sys.argv[1]
    assert feature in ('f10', 'f11')
    previous = 'f9' if feature == 'f10' else 'f10'
    baseline, _ = read_run('f9')
    before_metadata, before = read_run(previous)
    metadata, after = read_run(feature)
    for key in ('sha256', 'device', 'numpy_seed', 'torch_sampling_seed',
                'environment', 'python', 'numpy', 'torch', 'evaluation'):
        assert metadata[key] == baseline[key] == before_metadata[key], key
    # Prove the runners/evaluator are byte-identical to the captured F9 source.
    for name, digest in metadata['source_sha256'].items():
        if name != 'cfrbm/rbm.py':
            source = subprocess.check_output(
                ['git', 'show', f"{baseline['source_commit']}:{name}"], cwd=ROOT)
            assert hashlib.sha256(source).hexdigest() == digest, name
        if len(sys.argv) > 2:
            source = subprocess.check_output(['git', 'show', f'{sys.argv[2]}:{name}'], cwd=ROOT)
            assert hashlib.sha256(source).hexdigest() == digest, name
    summary = dict(
        feature=feature, before=previous,
        before_final_rmse=before[-1]['rmse'], after_final_rmse=after[-1]['rmse'],
        final_rmse_delta=after[-1]['rmse'] - before[-1]['rmse'],
        before_final_mae=before[-1]['mae'], after_final_mae=after[-1]['mae'],
        before_best_rmse=min(row['rmse'] for row in before),
        after_best_rmse=min(row['rmse'] for row in after),
        after_best_epoch=min(after, key=lambda row: row['rmse'])['iteration'] + 1,
        scored_ratings=metadata['evaluation']['scored_ratings'],
    )
    print('PASS: 100 epochs, finite metrics, log/JSON agreement, inputs, seeds, '
          'environment and unchanged evaluator match F9')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
