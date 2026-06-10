"""
gui.py

Contains the Tkinter-based graphical user interface for the Brand Cleaner application.
It delegates all data processing and file operations to the processor module.
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

from src.processor import (
    DEFAULT_SIMILARITY,
    clean_file_with_brands,
    load_clean_brands,
    read_single_column_series,
    write_single_column_series
)

APP_TITLE = "Brand Cleaner"

class BrandCleanerGUI(tk.Tk):
    """Main Application Window for the Brand Cleaner."""
    
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("820x560")
        self.minsize(760, 520)
        try:
            self.iconbitmap(default="")
        except Exception:
            pass

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.clean_brand_path_var = tk.StringVar()
        self.file_to_clean_path_var = tk.StringVar()
        self.similarity_var = tk.IntVar(value=DEFAULT_SIMILARITY)
        self.keep_original_var = tk.BooleanVar(value=True)

        self.add_clean_brand_path_var = tk.StringVar()
        self.new_brand_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_clean = ttk.Frame(notebook)
        notebook.add(frame_clean, text="Clean File")

        row = 0
        ttk.Label(frame_clean, text='Clean Brand names file (Excel/CSV):').grid(row=row, column=0, sticky="w", padx=6, pady=(8, 4))
        ttk.Entry(frame_clean, textvariable=self.clean_brand_path_var, width=70).grid(row=row, column=1, sticky="we", padx=6, pady=(8, 4))
        ttk.Button(frame_clean, text="Browse...", command=self._browse_clean_brand).grid(row=row, column=2, sticky="e", padx=6, pady=(8, 4))

        row += 1
        ttk.Label(frame_clean, text='File to clean (Excel/CSV):').grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(frame_clean, textvariable=self.file_to_clean_path_var, width=70).grid(row=row, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(frame_clean, text="Browse...", command=self._browse_file_to_clean).grid(row=row, column=2, sticky="e", padx=6, pady=4)

        row += 1
        ttk.Checkbutton(frame_clean, text="Keep original brand column (as 'Original Brand')", variable=self.keep_original_var).grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=4)

        row += 1
        sim_frame = ttk.Frame(frame_clean)
        sim_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=6, pady=6)
        ttk.Label(sim_frame, text="Min similarity (%) for replacement:").pack(side=tk.LEFT)
        slider = ttk.Scale(sim_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.similarity_var, command=lambda e: self._update_similarity_label())
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.sim_label = ttk.Label(sim_frame, text=f"{self.similarity_var.get()}%")
        self.sim_label.pack(side=tk.LEFT)

        row += 1
        btn_frame = ttk.Frame(frame_clean)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=6, pady=8)
        ttk.Button(btn_frame, text="Run and Save...", command=self._run_and_save).pack(side=tk.LEFT, padx=(0, 6))
        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        row += 1
        ttk.Label(frame_clean, text="Log:").grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=(8, 4))
        self.log_text = tk.Text(frame_clean, height=14)
        self.log_text.grid(row=row + 1, column=0, columnspan=3, sticky="nsew", padx=6, pady=(0, 8))
        frame_clean.grid_columnconfigure(1, weight=1)
        frame_clean.grid_rowconfigure(row + 1, weight=1)

        frame_add = ttk.Frame(notebook)
        notebook.add(frame_add, text="Add Clean Brand")

        r = 0
        ttk.Label(frame_add, text='Clean Brand names file (Excel/CSV):').grid(row=r, column=0, sticky="w", padx=6, pady=(8, 4))
        ttk.Entry(frame_add, textvariable=self.add_clean_brand_path_var, width=70).grid(row=r, column=1, sticky="we", padx=6, pady=(8, 4))
        ttk.Button(frame_add, text="Browse...", command=self._browse_add_clean_brand).grid(row=r, column=2, sticky="e", padx=6, pady=(8, 4))

        r += 1
        ttk.Label(frame_add, text='New clean brand to add:').grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(frame_add, textvariable=self.new_brand_var, width=40).grid(row=r, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(frame_add, text="Add", command=self._add_new_brand).grid(row=r, column=2, sticky="e", padx=6, pady=4)

        frame_add.grid_columnconfigure(1, weight=1)

        footer = ttk.Frame(self)
        footer.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Label(footer, text="Tip: Matching is token-anchored and prefix-first; scores get a word-only boost (never reduced).", foreground="#555").pack(side=tk.LEFT)

    def _update_similarity_label(self):
        self.sim_label.config(text=f"{int(self.similarity_var.get())}%")

    def _browse_clean_brand(self):
        path = filedialog.askopenfilename(
            title="Select Clean Brand names file",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.xlsm *.xlsb *.csv *.txt"), ("All files", "*.*")]
        )
        if path:
            self.clean_brand_path_var.set(path)

    def _browse_file_to_clean(self):
        path = filedialog.askopenfilename(
            title="Select file to clean",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.xlsm *.xlsb *.csv *.txt"), ("All files", "*.*")]
        )
        if path:
            self.file_to_clean_path_var.set(path)

    def _browse_add_clean_brand(self):
        path = filedialog.askopenfilename(
            title="Select Clean Brand names file",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.xlsm *.xlsb *.csv *.txt"), ("All files", "*.*")]
        )
        if path:
            self.add_clean_brand_path_var.set(path)

    def log(self, message: str):
        """Appends a message to the GUI log area."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def _run_and_save(self):
        clean_path = self.clean_brand_path_var.get().strip()
        target_path = self.file_to_clean_path_var.get().strip()
        
        if not clean_path:
            messagebox.showerror("Missing file", "Please select the Clean Brand names file.")
            return
        if not target_path:
            messagebox.showerror("Missing file", "Please select the file to clean.")
            return
        if not os.path.isfile(clean_path):
            messagebox.showerror("File not found", f"Clean Brand names file not found:\n{clean_path}")
            return
        if not os.path.isfile(target_path):
            messagebox.showerror("File not found", f"File to clean not found:\n{target_path}")
            return

        default_out = os.path.splitext(target_path)[0] + "_cleaned.xlsx"
        save_path = filedialog.asksaveasfilename(
            title="Save cleaned file as",
            defaultextension=".xlsx",
            initialfile=os.path.basename(default_out),
            filetypes=[("Excel Workbook", "*.xlsx")]
        )
        if not save_path:
            return

        def worker():
            try:
                self.progress.start(10)
                self.log("Loading clean brand names...")
                clean_list = load_clean_brands(clean_path)
                self.log(f"Loaded {len(clean_list)} clean brands (after removing empties and duplicates).")

                self.log("Cleaning file...")
                df, mapping = clean_file_with_brands(
                    clean_brand_path=clean_path,
                    file_to_clean_path=target_path,
                    min_similarity=int(self.similarity_var.get()),
                    keep_original=self.keep_original_var.get()
                )

                df.to_excel(save_path, index=False, engine="openpyxl")
                threshold = int(self.similarity_var.get())
                replaced = sum(1 for v, (final, score, sugg) in mapping.items() if score >= float(threshold))
                unique_brand_count = len(mapping)
                self.log(f"Unique brands processed: {unique_brand_count}")
                self.log(f"Values replaced by clean brand (score >= {threshold}%): {replaced}")

                # List brand names that weren't replaced with their best suggestion and boosted score
                not_replaced_rows = []
                for v, (final, score, sugg) in mapping.items():
                    vn = str(v).strip()
                    if vn and score < float(threshold):
                        not_replaced_rows.append((vn, sugg, score))
                        
                if not_replaced_rows:
                    not_replaced_rows.sort(key=lambda x: x[0].lower())
                    self.log(f"Brands not replaced ({len(not_replaced_rows)}):")
                    for orig, suggestion, score in not_replaced_rows:
                        self.log(f" - {orig} -> {suggestion} ({int(round(score))}%)")
                else:
                    self.log("Brands not replaced (0): - (none)")

                self.log(f"Saved cleaned file: {save_path}")
                messagebox.showinfo("Done", f"Cleaning complete!\nSaved: {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred:\n{e}")
                self.log(f"ERROR: {e}")
            finally:
                self.progress.stop()

        threading.Thread(target=worker, daemon=True).start()

    def _add_new_brand(self):
        path = self.add_clean_brand_path_var.get().strip()
        brand = self.new_brand_var.get().strip()
        if not path:
            messagebox.showerror("Missing file", "Please select the Clean Brand names file.")
            return
        if not brand:
            messagebox.showerror("Missing brand", "Please enter a brand to add.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("File not found", f"Clean Brand names file not found:\n{path}")
            return

        try:
            s = read_single_column_series(path)
            existing = [str(x).strip() for x in s.dropna().tolist() if str(x).strip()]
            exists_ci = {x.casefold(): x for x in existing}
            if brand.casefold() in exists_ci:
                messagebox.showinfo("No change", f"'{brand}' already exists in the clean brand list.")
                return

            new_series = pd.Series(existing + [brand])
            write_single_column_series(path, new_series, header=(s.name if s.name else "brand"))
            messagebox.showinfo("Success", f"Added '{brand}' to:\n{path}")
            self.new_brand_var.set("")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add brand:\n{e}")
