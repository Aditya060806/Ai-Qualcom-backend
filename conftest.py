"""Pytest bootstrap: put the project root on sys.path and FORCE mock mode so the
test suite is hermetic -- it must not depend on (or be broken by) ambient
BRUTUS_* environment variables left over from a real-mode run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force mock mode and clear any backend overrides inherited from the shell.
os.environ["BRUTUS_MODE"] = "mock"
for _key in ("BRUTUS_LLM_BACKEND", "BRUTUS_ASR_BACKEND", "BRUTUS_TTS_BACKEND"):
    os.environ.pop(_key, None)
