import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ============================
# 1. Load Data
# ============================

def load_data(path):
    """
    Load CSV exactly as-is with no column assumptions or conversions.
    """
    df = pd.read_csv(path)
    return df


# ============================
# 2. Basic Summary Stats
# ============================

def print_basic_stats(df):
    print("\n===== DATA OVERVIEW =====")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    print("===== SUMMARY OF NUMERIC COLUMNS =====")
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No numeric columns found.\n")


# ============================
# 3. Example Graphs (Optional)
# ============================

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
    
def clean_data(df):
    """
    Clean the DataFrame and return a cleaned copy.

    Behavior:
    - If the DataFrame has a 'dnf' column, drop any rows where its
      value is the literal string 'true' (case-insensitive).
    - If 'dnf' column is not present, return the DataFrame unchanged.
    """
    if df is None:
        return df

    # If there's no 'dnf' column, nothing to do
    if 'dnf' not in df.columns:
        return df

    # Convert values to string, compare lowercase to 'true' to be robust
    mask = df['dnf'].astype(str).str.lower() == 'true'

    # Keep rows where mask is False (i.e., not a DNF)
    cleaned_df = df.loc[~mask].copy()
    return cleaned_df


# ============================
# 4. Main Execution
# ============================

def main():
    file_path = "solves (3).csv"  # <-- your CSV path

    df = load_data(file_path)
    df = clean_data(df)

    print_basic_stats(df)
    


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