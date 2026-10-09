"""Check the captured F9 baseline without rerunning training (standard library only)."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import re


def main():
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[1]
    config = json.loads((root / 'ubased.json').read_text())['experiments'][0]['configs'][0]
    result_path = evidence / 'initial_experiment_100k-u1xu1.json'
    result = json.loads(result_path.read_text())
    metadata = json.loads((evidence / 'run.json').read_text())
    assert metadata['exit_code'] == 0, 'Training did not finish successfully'
    for name, digest in metadata['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    train_rows = [line.split('\t') for line in (root / 'ml-100k/u1.base').read_text().splitlines()]
    test_rows = [line.split('\t') for line in (root / 'ml-100k/u1.test').read_text().splitlines()]
    movies = {row[1] for row in train_rows}
    tests = defaultdict(list)
    for user, movie, *_ in test_rows:
        tests[user].append(movie)
    scored = 0
    for user_movies in tests.values():
        for movie in user_movies:
            if movie not in movies:
                break  # Match the existing runner's per-user exception boundary.
            scored += 1
    coverage = metadata['evaluation']
    assert coverage['train_ratings'] == len(train_rows) == 80000
    assert coverage['train_users'] == len({row[0] for row in train_rows}) == 943
    assert coverage['train_movies'] == len(movies) == 1650
    assert coverage['test_ratings'] == len(test_rows) == 20000
    assert coverage['scored_ratings'] == scored == 19837
    assert coverage['skipped_ratings'] == len(test_rows) - scored == 163
    assert coverage['unknown_movie_ratings'] == sum(row[1] not in movies for row in test_rows) == 32
    assert {key: value for key, value in result.items() if key != 'results'} == config
    rows = result['results']
    assert config['epochs'] == len(rows) == 100, 'Expected all 100 epochs'
    assert [row['iteration'] for row in rows] == list(range(100))
    log_rows = re.findall(
        r'^End iter (\d+) - k/lr: (\d+)/([\d.e+-]+) momentum: ([\d.e+-]+) '
        r'- MAE/RMSE: ([\d.e+-]+)/([\d.e+-]+)$',
        (evidence / 'training.log').read_text(), re.MULTILINE,
    )
    assert len(log_rows) == 100, 'Expected 100 console metric records'
    for row, logged in zip(rows, log_rows):
        assert all(math.isfinite(row[key]) for key in ('mae', 'rmse'))
        assert 0 <= row['mae'] <= row['rmse'] <= 4
        assert row['k'] == 1 and row['lrate'] == 0.0005
        assert row['momentum'] == (0.5 if row['iteration'] < 50 else 0.6)
        expected = tuple(row[key] for key in ('iteration', 'k', 'lrate', 'momentum', 'mae', 'rmse'))
        assert tuple(map(float, logged)) == expected, 'Console/JSON mismatch'
    print(f"PASS: 100 epochs, matching console/JSON, final MAE {rows[-1]['mae']:.6f}, "
          f"RMSE {rows[-1]['rmse']:.6f}; config, input hashes and coverage match")
    if not 0.95 <= rows[-1]['rmse'] <= 1.0:
        raise SystemExit('FAIL: final RMSE outside acceptance range 0.95–1.0')
    print('PASS: final RMSE within acceptance range 0.95–1.0')


if __name__ == '__main__':
    main()
