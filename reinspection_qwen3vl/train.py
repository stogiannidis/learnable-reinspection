"""Compatibility entrypoint — prefer ``python -m reinspection_vlm.train --backend qwen3vl``."""

import sys


def main():
    argv = [sys.argv[0]]
    if "--backend" not in sys.argv:
        argv.extend(["--backend", "qwen3vl"])
    argv.extend(sys.argv[1:])
    sys.argv = argv
    from reinspection_vlm.train import main as _main

    _main()


if __name__ == "__main__":
    main()
