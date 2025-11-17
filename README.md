# Cubeast Analysis

IMPORTANT NOTE: this will currently only work properly if using cubeast `CFOP (2 look OLL)` method. If using `CFOP`, will basically work but PLL graph title will say "Corner OLL" I believe.
Tools and scripts to analyze cube solves and generate interactive plots given cubeast export. 

## Overview
This repository contains scripts to load solves data (CSV), clean the data, and generate both static Matplotlib plots and interactive Plotly visualizations. The interactive plots are saved as HTML files and served locally so you can view all plots on a single webpage.

Files of interest
- `import.py` - main runner: loads data, cleans it, generates interactive plots, and serves them on `http://localhost:8000`.
- `data.py` - data loading and cleaning helpers (`load_data`, `clean_data`).
- `plots.py` - plotting helpers:
  - Matplotlib functions: `graph_column`, `plot_histogram`, etc.
  - Interactive helpers: `fig_from_series`, `save_interactive_plots`, `serve_interactive` (use Plotly).
- `.gitignore` - ignores generated files including `interactive_plots/`.

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- plotly (for interactive HTML export)

Install dependencies with pip using the provided `requirements.txt`:

```powershell
pip install -r requirements.txt
```

`requirements.txt` is included in the repository and lists the runtime dependencies. Edit it to pin specific versions if you want reproducible installs.

## Usage
Place your CSV (example `solves.csv`) in the repository root.

Run the main script:

```powershell
python import.py
```

Behavior:
- The script will load and clean the data (drops rows where `dnf` == `'true'`).
- It will generate interactive Plotly HTML files in `interactive_plots/` and a combined `interactive_plots/index.html`.
- It will start a simple HTTP server and open your browser to `http://localhost:8000/index.html`.

