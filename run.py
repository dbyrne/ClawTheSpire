#!/usr/bin/env python
"""Launch the STS2 autonomous runner.

Usage:
    python run.py              # full auto
    python run.py --step       # step through each action
    python run.py --dry-run    # show decisions without executing
"""

from sts2_solver.runner import main

if __name__ == "__main__":
    main()
