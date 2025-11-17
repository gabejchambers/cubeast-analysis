import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
import webbrowser
import http.server
import socketserver
import threading

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False


# Map of scatterable columns to descriptive titles (module-level)
# Keys are column names in the CSV; values are the exact titles to use
SCATTERABLE_COLS = {
    'time' : 'Total Solve Time',
    'total_recognition_time': 'Total Recognition Time',
    'total_execution_time': 'Total Execution Time',
    'step_0_execution_time': 'Cross Execution Time',
    'step_1_time': 'F2L1 Time',
    'step_1_recognition_time': 'F2L1 Recognition Time',
    'step_1_execution_time': 'F2L1 Execution Time',
    'step_2_time': 'F2L2 Time',
    'step_2_recognition_time': 'F2L2 Recognition Time',
    'step_2_execution_time': 'F2L2 Execution Time',
    'step_3_time': 'F2L3 Time',
    'step_3_recognition_time': 'F2L3 Recognition Time',
    'step_3_execution_time': 'F2L3 Execution Time',
    'step_4_time': 'F2L4 Time',
    'step_4_recognition_time': 'F2L4 Recognition Time',
    'step_4_execution_time': 'F2L4  Execution Time',
    'step_4_cumulative_time': 'Cross + F2L Time',
    'step_5_time': 'Edge OLL Time',
    'step_5_recognition_time': 'Edge OLL Recognition Time',
    'step_5_execution_time': 'Edge OLL Execution Time',
    'step_6_time': 'Corner OLL Time',
    'step_6_recognition_time': 'Corner OLL Recognition Time',
    'step_6_execution_time': 'Corner OLL Execution Time',
    'step_6_cumulative_time': 'Cross + F2L + OLL Time',
    'step_7_time': 'PLL Time',
    'step_7_recognition_time': 'PLL Recognition Time',
    'step_7_execution_time': 'PLL Execution Time'
}


def plot_first_numeric_series(df):
    """
    Plot the first numeric column found.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        print("No numeric columns available for plotting.")
        return

    col = numeric_cols[0]

    plt.figure(figsize=(10, 4))
    plt.plot(df[col])
    plt.title(f"Plot of {col}")
    plt.xlabel("Row")
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histogram(df):
    """
    Plot histogram of first numeric column found.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        print("No numeric columns available for histogram.")
        return

    col = numeric_cols[0]

    plt.figure(figsize=(8, 4))
    plt.hist(df[col].dropna(), bins=40)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def graph_column(df, column, convert_ms=True, show=True):
    """
    Create a scatter plot for a given numeric column from df.

    Parameters:
    - df: pandas DataFrame
    - column: column name (string)
    - convert_ms: if True, divide values by 1000 to convert ms -> seconds
    - show: if True, call plt.show() (useful to disable when testing)

    The x-axis is the row index (0..n-1) for the selected data points.
    A line of best fit is added using numpy.polyfit.
    """
    if df is None:
        raise ValueError("df must be a DataFrame")

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    # Extract values and coerce to numeric, dropping NA
    series = pd.to_numeric(df[column], errors='coerce').dropna()
    if len(series) == 0:
        raise ValueError(f"No numeric data found in column '{column}'")

    y = series.astype(float).values
    if convert_ms:
        y = y / 1000.0

    x = np.arange(len(y))

    plt.figure(figsize=(10, 4))
    plt.scatter(x, y, label=column, alpha=0.6)

    # Best-fit line (degree 1) using numpy.polyfit
    coeffs = np.polyfit(x, y, 1)
    fit_y = np.polyval(coeffs, x)
    plt.plot(x, fit_y, color='red', linewidth=2, label=f'Best fit: y={coeffs[0]:.4f}x+{coeffs[1]:.4f}')
    
    # show the change over the data set in line of best fit (how much better or worse you've gotten)
    n = len(x)
    delta = coeffs[0] * (n - 1)   # units = same units as y (seconds if converted)
    # Add the best-fit change as a legend entry so it won't be hidden by the plot
    # Create an invisible artist with the label text; it will appear in the legend
    plt.plot([], [], label=f"Best-fit change: {delta:.2f} s")

    # Use the mapping value as the exact title (fall back to column name)
    title = SCATTERABLE_COLS.get(column, column)

    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel(f"{title} (seconds)" if convert_ms else title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save using the column name as filename (sanitize for OS)
    safe_name = re.sub(r'[<>:\\"/\\|?*]', '', str(column))
    safe_name = safe_name.replace(' ', '_')
    filename = f"{safe_name}.png"
    file_path = os.path.join(os.getcwd(), filename)
    try:
        plt.savefig(file_path, dpi=200, bbox_inches='tight')
    except Exception:
        # don't raise on save errors
        pass

    if show:
        plt.show()


def fig_from_series(series, title, convert_ms=True):
    """Return a Plotly Figure for given pandas Series (convert_ms optional)."""
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is required for interactive figures. Install with `pip install plotly`.")

    y = pd.to_numeric(series, errors='coerce').dropna().astype(float).values
    if convert_ms:
        y = y / 1000.0

    x = np.arange(len(y))
    if len(x) == 0:
        raise ValueError("No numeric data to plot")

    # best-fit line
    coeffs = np.polyfit(x, y, 1)
    fit_y = np.polyval(coeffs, x)
    delta = coeffs[0] * (len(x) - 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='data', marker=dict(opacity=0.7)))
    fig.add_trace(go.Scatter(x=x, y=fit_y, mode='lines', name=f'Best fit: y={coeffs[0]:.4f}x+{coeffs[1]:.4f}', line=dict(color='red')))

    # annotation for delta in top-left (paper coords)
    fig.update_layout(
        title=title,
        xaxis_title='Index',
        yaxis_title='Time (s)' if convert_ms else 'Value',
        annotations=[dict(text=f"Best-fit change: {delta:.2f} s", x=0.01, y=0.99, xref='paper', yref='paper', showarrow=False, bgcolor='white')]
    )

    return fig


def save_interactive_plots(df, out_dir='interactive_plots', convert_ms=True, save_individual=True):
    """Save interactive Plotly HTMLs for all columns in SCATTERABLE_COLS and produce a single index.html.

    - `out_dir`: folder to write HTML files
    - `save_individual`: if True, write per-column full HTML files; always writes combined `index.html`.
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is required for interactive plots. Install with `pip install plotly`.")

    os.makedirs(out_dir, exist_ok=True)

    fragments = []
    for col, title in SCATTERABLE_COLS.items():
        if col not in df.columns:
            continue
        series = df[col]
        try:
            fig = fig_from_series(series, title, convert_ms=convert_ms)
        except Exception as e:
            print(f"Skipping {col}: {e}")
            continue

        safe_name = re.sub(r'[<>:\\"/\\|?*]', '', str(col)).replace(' ', '_')
        if save_individual:
            individual_path = os.path.join(out_dir, f"{safe_name}.html")
            fig.write_html(individual_path, full_html=True, include_plotlyjs='cdn')

        # append fragment (include CDN for safety so each fragment is self-contained)
        fragment = fig.to_html(full_html=False, include_plotlyjs='cdn')
        fragments.append(f"<h2>{title}</h2>\n" + fragment)

    if len(fragments) == 0:
        raise RuntimeError("No interactive plots were generated (no matching columns found).")

    index_path = os.path.join(out_dir, 'index.html')
    with open(index_path, 'w', encoding='utf8') as fh:
        fh.write('<!doctype html>\n<html>\n<head>\n<meta charset="utf-8"/>\n')
        fh.write('<title>Interactive Plots</title>\n</head>\n<body>\n')
        fh.write('<div style="max-width:1000px;margin:auto;">\n')
        fh.write('\n<hr/>\n'.join(fragments))
        fh.write('\n</div>\n</body>\n</html>')

    return index_path


def serve_interactive(out_dir='interactive_plots', port=8000, open_browser=True):
    """Serve the `out_dir` folder on localhost using a simple HTTP server and open the index in a browser.

    This function blocks until interrupted (Ctrl+C).
    """
    index_path = os.path.join(os.getcwd(), out_dir, 'index.html')
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"{index_path} not found. Run save_interactive_plots first.")

    # serve from out_dir
    handler = http.server.SimpleHTTPRequestHandler
    os.chdir(out_dir)
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/index.html"
        if open_browser:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        print(f"Serving {out_dir} at {url} (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")
