# Ensure 'src/' is on sys.path for tests so imports like 'from core...' work
import os
import sys

PROJECT_ROOT = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
