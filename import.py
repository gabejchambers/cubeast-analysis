from data import load_data, clean_data
from plots import graph_column, plot_histogram, SCATTERABLE_COLS
from plots import save_interactive_plots, serve_interactive


# ============================
# 4. Main Execution
# ============================

def main():
    file_path = "solves (3).csv"  # <-- your CSV path

    df = load_data(file_path)
    df = clean_data(df)
    
    # Interactive export/serve will be attempted below; do not show
    # Matplotlib popups here to avoid opening plots one-by-one.

    # Try to save interactive plots and serve them on localhost
    try:
        out_index = save_interactive_plots(df, out_dir='interactive_plots', convert_ms=True, save_individual=True)
        print(f"Interactive plots written. Open {out_index} or serving locally...")
        # Serve the interactive index - this will block until Ctrl+C
        serve_interactive(out_dir='interactive_plots', port=8000, open_browser=True)
    except Exception as e:
        print(f"Interactive export/serve failed: {e}")
        # fallback to static/matplotlib behavior
        for col in SCATTERABLE_COLS:
            try:
                graph_column(df, col)
            except Exception as e2:
                print(f"Could not plot '{col}' column: {e2}")



if __name__ == "__main__":
    main()