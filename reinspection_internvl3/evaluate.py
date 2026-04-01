"""Compatibility entrypoint — prefer ``python -m reinspection_vlm.evaluate --backend internvl3``."""

import sys


def main():
    argv = [sys.argv[0]]
    if "--backend" not in sys.argv:
        argv.extend(["--backend", "internvl3"])
    argv.extend(sys.argv[1:])
    sys.argv = argv
    from reinspection_vlm.evaluate import main as _main

    _main()


if __name__ == "__main__":
    main()
