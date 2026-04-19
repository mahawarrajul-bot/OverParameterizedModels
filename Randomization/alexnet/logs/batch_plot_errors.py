#!/usr/bin/env python3
"""Simple wrapper to plot errors for all corruption levels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from plot_single_corr import plot_error_vs_epochs

log_dir = Path(__file__).parent
corruptions = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]

for c in corruptions:
    try:
        print(f"Generating error plot for corruption {c:.1f}...", end=" ")
        plot_error_vs_epochs(c, log_dir=str(log_dir))
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\nAll plots generated!")
