# Brand Cleaner

A robust, portfolio-ready application featuring a sophisticated fuzzy matching engine and a clean Tkinter graphical user interface (GUI) for data standardization.

This tool is designed to clean and standardise brand names from messy datasets (like sales reports or product catalogs) by matching them against a clean, pre-approved list of brands.

## Project Architecture

The codebase has been structured with clear separation of concerns, demonstrating software engineering best practices:

- **`src/matcher.py`**: Contains the core algorithmic engine for fuzzy string matching, tokenization, and complex string similarity scoring.
- **`src/processor.py`**: Handles Pandas DataFrame manipulation, file I/O operations (Excel/CSV), and applies the matching logic to the datasets.
- **`src/gui.py`**: Contains the Tkinter GUI implementation, delegating data operations to the processor module.
- **`main.py`**: The clean entry point to start the application.
- **`tests/`**: Unit tests verifying the string matching algorithms and data processing functions.

## Features
- Add new clean brands to your Clean Brand file (deduplicated, case-insensitive).
- Clean any file where the first column is the brand column:
  - Removes trailing `uk` (case-insensitive) from brand names.
  - Fuzzy-matches against your Clean Brand list and replaces when similarity meets your threshold.
  - Keeps original brand column optionally.
  - Adds a `Group code` column using `tuck_<alphanumeric brand>`.
- Adjustable similarity threshold (0–100). Default is 80.
- Works with Excel (`.xlsx`, `.xls`, `.xlsm`, `.xlsb`) and CSV (`.csv`, `.txt`).
- Saves cleaned output as an Excel workbook (`.xlsx`).

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python main.py
   ```

## Running Tests

This project includes a suite of unit tests written with `pytest` to ensure the reliability of the matching algorithms.

```bash
# Ensure you have activated the virtual environment and installed requirements
pytest tests/
```

## Usage Tips

- The Clean Brand names file must have a single meaningful column of brand names. If it has multiple columns, only the first is used.
- Empty rows in the Clean Brand names file are ignored automatically.
- The file to be cleaned must have the brand names in its first column.
- The app removes trailing `uk` from brand names in the file to be cleaned before matching (e.g., `Acme UK`, `Acme-uk`, `Acme_uk` → `Acme`).
- If the best similarity score for a brand is below your chosen threshold, the brand stays as-is (after the `uk` removal step).
- Group code is generated as `tuck_<brandname>`, where `<brandname>` is the final brand value with all non-alphanumeric characters removed and converted to lowercase.

## Packaging as a Desktop App (PyInstaller)

You can create a single-file executable for Windows or macOS.

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

2. Build:
   ```bash
   pyinstaller --onefile --noconsole --name "BrandCleaner" main.py
   ```

3. Find the executable in the `dist` folder:
   - Windows: `dist/BrandCleaner.exe`
   - macOS: `dist/BrandCleaner`

4. macOS Gatekeeper note:
   - The app may be flagged as from an unidentified developer. Right-click the app, choose “Open”, then confirm.
   - For distribution, consider code-signing and notarization (optional).

## Troubleshooting

- If `.xlsx` files fail to load or save, ensure `openpyxl` is installed (it is listed in `requirements.txt`).
- If fuzzy matching seems slow or inaccurate, make sure `rapidfuzz` installed correctly. The app will fall back to Python’s `difflib` if `rapidfuzz` is unavailable.
- If your files are very large, the first run may take a bit; the UI stays responsive while processing.

## Notes

- The app replaces the first column values with matched Clean Brand names only when the similarity score meets your threshold; otherwise it keeps the cleaned original.
- Set a lower threshold (e.g., 50%) if you want more aggressive replacement.