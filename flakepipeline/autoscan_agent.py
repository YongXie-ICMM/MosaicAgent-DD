"""Compatibility entry for offline focus checks only.

This publication package does not include the historical stage-control GUI in
this module. Use the separately documented acquisition/Auto_Scan application
for instrument operation. Importing or invoking this file never opens hardware.
"""
import sys

if __package__:
    from .focus_metrics import *  # noqa: F401,F403
    from .focus_metrics import main
else:
    from focus_metrics import *  # noqa: F401,F403
    from focus_metrics import main

if __name__ == "__main__":
    sys.exit(main())
