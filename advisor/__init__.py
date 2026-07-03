"""ADvisor package entry points."""


def main(*args, **kwargs):
    from importlib import import_module

    cli_main = import_module("main").main

    return cli_main(*args, **kwargs)

__all__ = ["main"]
