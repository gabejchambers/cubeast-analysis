from data import load_data, clean_data
from plots import graph_column, plot_histogram, SCATTERABLE_COLS


# ============================
# 4. Main Execution
# ============================

def main():
    file_path = "solves (3).csv"  # <-- your CSV path

    df = load_data(file_path)
    df = clean_data(df)
    
    # Optional plots
    # Plot total solve time (assumes a 'time' column in ms)
    for col in SCATTERABLE_COLS:
        try:
            graph_column(df, col)
        except Exception as e:
            print(f"Could not plot '{col}' column: {e}")

    plot_histogram(df)


if __name__ == "__main__":
    main()