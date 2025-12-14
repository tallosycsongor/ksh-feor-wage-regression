KSH STADAT – Wage Trend Analysis (Python)
=========================================
This project uses publicly available KSH (Hungarian Central Statistical Office) STADAT data to generate a small statistical analysis and visualizations.
What the program does:
- Reads a downloaded CSV/XLSX dataset
- Builds a clean time series
- Prints basic statistics
- Creates a line chart (trend)
- Creates a scatter plot
- Fits a linear regression on the scatter plot (equation + R²)
Data source
-----------
KSH STADAT table: mun0208 – Gross average earnings of full-time employees by occupation (FEOR’08), 2019–2024.
Download links:
- CSV: https://www.ksh.hu/stadat_files/mun/hu/mun0208.csv
- XLSX: https://www.ksh.hu/stadat_files/mun/hu/mun0208.xlsx
Place the downloaded file into the project folder before running the script.
Requirements
------------
- Python 3.10+ (recommended)
- Libraries: pandas, matplotlib, scikit-learn (for Excel input: openpyxl)
Install
-------
pip install -r requirements.txt
Run
---
1) Put the KSH file into the project directory (e.g., mun0208.csv).
2) Start the program:
python main.py
The script asks for a FEOR code. If you press Enter, it uses the default code.
Outputs
-------
After running, an output/ folder is created with:
- output/tisztitott_idosor.csv – processed time series (year, value, YoY)
- output/vonaldiagram.png – line chart
- output/pontdiagram_regresszio.png – scatter plot + regression line + equation + R²
The console output includes:
- basic statistics (min, max, mean, std, total change)
- year-over-year change (absolute + %)
- regression slope (Ft/year) and R² score
Interpretation
--------------
- The line chart shows how gross average earnings changed year by year for a selected occupation group.
- YoY change highlights which years increased faster or slower.
- Linear regression gives a simple trend estimate: slope = average growth in Ft per year; R² = fit quality (closer to 1 = better).
Note: Values are nominal gross earnings (not inflation-adjusted).
Development notes / challenges
------------------------------
KSH CSV values often contain spaces as thousands separators (e.g., "1 825 697") and may include missing values ("–"), so cleaning/parsing is required after reading.
