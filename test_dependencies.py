#!/usr/bin/env python3
"""
Test script to verify all dependencies are available
"""

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    print("✓ plotly imported successfully")
except ImportError as e:
    print(f"✗ plotly import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import json
    import os
    from pathlib import Path
    print("✓ standard library imports successful")
except ImportError as e:
    print(f"✗ standard library import failed: {e}")

print("All dependency checks completed!")
