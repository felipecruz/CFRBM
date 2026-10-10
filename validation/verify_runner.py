"""Compare F14 with the pre-consolidation runners on MovieLens u1.

Run with PYTHONHASHSEED=17 PYTHONPATH=. and redirect stdout to evidence JSON.
The non-default item baseline changes only its two hardcoded batch sizes.
"""

import json
import platform
import subprocess
import types

import numpy as np
import torch

from cfrbm import runner
from cfrbm.dataset import load_dataset
from validation.benchmark_preparation import ROOT, digest, runner_trial


def main():
    baseline_commit = '9e8fa056926bdf7087b8bc92bdf5da3b181527bf'
    torch.set_num_threads(1)
    config = json.loads((ROOT / 'ubased.json').read_text())['experiments'][0]['configs'][0]
    config = dict(config, epochs=3)
    report = dict(baseline_commit=baseline_commit, python=platform.python_version(),
                  numpy=np.__version__, torch=torch.__version__, device='cpu', seed=17,
                  torch_threads=1, epochs=3,
                  input_sha256={p: digest(ROOT / p) for p in
                                ('ml-100k/u1.base', 'ml-100k/u1.test', 'ubased.json')},
                  source_sha256={str(p.relative_to(ROOT)): digest(p)
                                 for p in sorted((ROOT / 'cfrbm').glob('*.py'))}, comparisons=[])
    for mode in ('user', 'item'):
        users, movies, tests = load_dataset(ROOT / 'ml-100k/u1.base',
                                           ROOT / 'ml-100k/u1.test', '\t', mode == 'user')
        source = subprocess.check_output(
            ['git', 'show', f'{baseline_commit}:cfrbm/{mode}_based.py'], cwd=ROOT, text=True)
        for size in (10, 7):
            adjusted = mode == 'item' and size != 10
            baseline_source = source
            if adjusted:
                assert source.count('batch_size = 10') == 2
                baseline_source = source.replace('batch_size = 10', "batch_size = config['batch_size']")
            baseline = types.ModuleType(f'cfrbm._baseline_{mode}')
            baseline.__package__ = 'cfrbm'
            exec(compile(baseline_source, f'<baseline-{mode}>', 'exec'), baseline.__dict__)
            trial_config = dict(config, batch_size=size)
            before = runner_trial(baseline, users, movies, tests, trial_config)
            after = runner_trial(runner, users, movies, tests, trial_config, orientation=mode)
            for key in ('metrics', 'state_sha256', 'batches_sha256'):
                assert before[key] == after[key], (mode, size, key)
            report['comparisons'].append(dict(orientation=mode, batch_size=size,
                                              baseline_batch_size_corrected=adjusted,
                                              exact_parity=True, before=before, after=after))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
