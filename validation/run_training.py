"""Capture a seed-17 CPU run in an isolated source snapshot.

Usage: .venv/bin/python validation/run_training.py f10
The snapshot prevents subsequent working-tree edits or existing root results
from affecting a running experiment. No training or evaluation math is changed.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from datetime import datetime, timezone

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT = 'initial_experiment_100k-u1xu1.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    feature = sys.argv[1]
    if feature not in ('f10', 'f11', 'f12'):
        raise SystemExit('Expected f10, f11 or f12')
    output = ROOT / 'validation' / feature
    output.mkdir(exist_ok=True)
    if (output / 'run.json').exists():
        raise SystemExit('Evidence already exists; choose a clean checkout to rerun')
    environment = dict(PYTHONHASHSEED='17', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    wrapper = ('import runpy, sys, numpy as np; np.random.seed(17); '
               'sys.argv = ["cfrbm.user_based", "ubased.json", "--device", "cpu"]; '
               'runpy.run_module("cfrbm.user_based", run_name="__main__")')
    with TemporaryDirectory(prefix=f'cfrbm-{feature}-') as directory:
        snapshot = Path(directory)
        shutil.copytree(ROOT / 'cfrbm', snapshot / 'cfrbm',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT / 'ubased.json', snapshot / 'ubased.json')
        (snapshot / 'ml-100k').symlink_to(ROOT / 'ml-100k', target_is_directory=True)
        metadata = dict(
            feature=feature, parent_commit=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            source_sha256={str(p.relative_to(snapshot)): digest(p)
                           for p in sorted((snapshot / 'cfrbm').glob('*.py'))},
            sha256={name: digest(snapshot / name) for name in
                    ['ubased.json', 'ml-100k/u1.base', 'ml-100k/u1.test']},
            python=platform.python_version(), platform=platform.platform(),
            numpy=np.__version__, torch=torch.__version__, device='cpu',
            numpy_seed=17, torch_sampling_seed=17, environment=environment,
            command=[sys.executable, '-u', '-c', wrapper],
            evaluation=json.loads((ROOT / 'validation/f9/run.json').read_text())['evaluation'],
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        started = time.monotonic()
        # Save the source fingerprint immediately so progress is attributable.
        (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
        with (output / 'training.log').open('w') as log:
            process = subprocess.run(metadata['command'], cwd=snapshot,
                                     env={**os.environ, **environment},
                                     stdout=log, stderr=subprocess.STDOUT)
        metadata.update(exit_code=process.returncode,
                        elapsed_seconds=time.monotonic() - started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        if (snapshot / RESULT).exists():
            shutil.copy2(snapshot / RESULT, output / RESULT)
        (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(metadata, indent=2))
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
