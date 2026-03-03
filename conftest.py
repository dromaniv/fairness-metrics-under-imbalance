# conftest.py – makes the repo root importable when running pytest from any directory.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
