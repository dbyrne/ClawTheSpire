"""sts2_solver package."""

# Reconfigure stdout/stderr to UTF-8 on import. Windows defaults to cp1252
# ("charmap"), which crashes when the codebase prints intentional Unicode
# (em-dashes in headers, arrows in eval output, etc.). Each Python process
# importing sts2_solver gets UTF-8 stdout for free, so we don't need
# PYTHONIOENCODING=utf-8 prefixes on every command or sys.stdout.reconfigure
# at every entry point. See feedback_windows_utf8_permanent_fix.md.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Detached, non-tty, or already utf-8 — nothing to do.
        pass
