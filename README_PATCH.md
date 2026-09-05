# Phase 8 runtime import hotfix

This patch changes the Phase 8 smoke-test invocation from direct script execution to module execution:

```bash
python -m scripts.run_phase8
```

Running the script as a module keeps the repository root on Python's import path, so the root-level `api` package resolves reliably. It also adds `scripts/__init__.py` to make the module boundary explicit.
