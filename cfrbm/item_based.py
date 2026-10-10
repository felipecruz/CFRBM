"""Compatibility entry point for item-based experiments."""

from .runner import main, run as _run


def run(name, dataset, config, all_users, all_movies, tests, initial_v, sep, device=None):
    return _run(name, dataset, config, all_users, all_movies, tests, initial_v,
                sep, device=device, orientation='item')


if __name__ == '__main__':
    main(orientation='item')
