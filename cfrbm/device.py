"""Shared device selection and command-line options for both runners."""

import argparse

import torch


def select_device(device=None):
    """Resolve auto as CUDA, MPS, CPU; reject unavailable explicit choices."""
    if device is None or str(device) == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    try:
        selected = torch.device(device)
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValueError(f'Invalid device {device!r}; use auto, cpu, mps or cuda[:index]') from error

    if selected.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available; use --device cpu or another available device')
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index {index} is not available')
        return torch.device('cuda', index)
    if selected.type in ('cpu', 'mps'):
        if selected.index not in (None, 0):
            raise ValueError(f'{selected.type.upper()} supports only device index 0')
        if selected.type == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is not available; use --device cpu or another available device')
        return torch.device(selected.type)
    raise ValueError(f'Unsupported device {device!r}; use auto, cpu, mps or cuda[:index]')


def parse_args(argv=None, *, orientation='user'):
    parser = argparse.ArgumentParser(description='Train a collaborative-filtering RBM')
    parser.add_argument('experiment', help='path to the experiment JSON file')
    parser.add_argument('--device', default='auto', metavar='DEVICE',
                        help='auto (default: CUDA, then MPS, then CPU), cpu, mps or cuda[:index]')
    parser.add_argument('--orientation', choices=('user', 'item'), default=orientation,
                        help='profile orientation (default: %(default)s)')
    args = parser.parse_args(argv)
    try:
        args.device = select_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    return args
