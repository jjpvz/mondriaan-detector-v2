import joblib
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import os, sys
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import csv
import json
import re
from functools import lru_cache


CSV_PATH = "C:\\GIT\\mondriaan-detector-v2\\texts.csv"  # <-- pas aan indien nodig


def _prediction_to_csv_label(prediction):
    """
    Model geeft 0..14 terug.
    texts.csv gebruikt label 1..15.
    Dus: csv_label = model_label + 1
    Niet-mondriaan is model 14 -> csv 15 (werkt ook voor string 'niet_mondriaan')
    """
    if prediction is None:
        return None

    if isinstance(prediction, int):
        return prediction + 1

    if isinstance(prediction, str):
        s = prediction.strip().lower()
        if s.isdigit():
            return int(s) + 1
        if s in {"niet_mondriaan", "niet mondriaan", "no_mondriaan", "non_mondrian"}:
            return 15

    return None


def _load_texts_csv(csv_path):
    """
    Leest texts.csv: label;name;description;table
    table is JSON-string (kan leeg zijn).
    Return: dict[int] -> dict
    """
    data = {}
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    last_error = None
    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    try:
                        lbl = int(str(row.get("label", "")).strip())
                    except ValueError:
                        continue

                    table_raw = (row.get("table") or "").strip()
                    table_obj = None
                    if table_raw:
                        try:
                            table_obj = json.loads(table_raw)
                        except json.JSONDecodeError:
                            table_obj = None

                    data[lbl] = {
                        "label": lbl,
                        "name": (row.get("name") or "").strip(),
                        "description": (row.get("description") or "").strip(),
                        "table": table_obj,
                    }
            return data
        except UnicodeDecodeError as exc:
            last_error = exc
            data = {}
            continue

    if last_error:
        raise last_error
    return data


def show_prediction_window(image, prediction, probability, auto_close_ms=None, csv_path=CSV_PATH):
    """
    Shows a GUI window with the image + texts.csv info for the predicted class.
    prediction: model output (0..14) or string.
    probability: float 0..1
    """
    texts = _load_texts_csv(csv_path)
    csv_label = _prediction_to_csv_label(prediction)
    info = texts.get(csv_label)

    # Fallbacks
    name = info["name"] if info and info["name"] else f"Klasse {prediction}"
    description = info["description"] if info and info["description"] else "Geen beschrijving gevonden in texts.csv."
    meta = info["table"] if info else None

    # Niet-mondriaan: model=14 -> csv=15
    is_negative = (csv_label == 15)

    # Kleuren
    ok_color = "#2E8B57"
    bad_color = "#DC143C"
    prediction_color = bad_color if is_negative else ok_color
    symbol = "✗" if is_negative else "✓"
    header_text = f"{symbol} {name}"
    confidence_text = f"Zekerheid: {probability * 100:.2f}%"

    # -------- GUI --------
    root = tk.Tk()
    root.title("Mondriaan Detector - Resultaat")
    root.geometry("1020x560")
    root.configure(bg="#f0f0f0")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("Title.TLabel", font=("Arial", 18, "bold"))
    style.configure("Header.TLabel", font=("Arial", 15, "bold"))
    style.configure("Small.TLabel", font=("Arial", 10))
    style.configure("Section.TLabel", font=("Arial", 10, "bold"))

    main = ttk.Frame(root, padding=18)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    main.columnconfigure(0, weight=0)  # image
    main.columnconfigure(1, weight=1)  # info
    main.rowconfigure(1, weight=1)

    title = ttk.Label(main, text="Mondriaan Detector", style="Title.TLabel")
    title.grid(row=0, column=0, columnspan=2, pady=(0, 14))

    # Image
    display_image = cv.resize(image, (340, 255))  # 4:3-ish
    display_image_rgb = cv.cvtColor(display_image, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(display_image_rgb)
    photo = ImageTk.PhotoImage(pil_image)

    image_card = ttk.LabelFrame(main, text="Foto", padding=10)
    image_card.grid(row=1, column=0, sticky="n", padx=(0, 14))

    image_label = ttk.Label(image_card, image=photo)
    image_label.grid(row=0, column=0)
    image_label.photo = photo

    # Info card
    info_card = ttk.LabelFrame(main, text="Resultaat", padding=14)
    info_card.grid(row=1, column=1, sticky="nsew")
    info_card.columnconfigure(0, weight=1)
    info_card.rowconfigure(4, weight=1)   # description grow
    info_card.rowconfigure(6, weight=1)   # meta grow

    header = ttk.Label(info_card, text=header_text, style="Header.TLabel", foreground=prediction_color)
    header.grid(row=0, column=0, sticky="w", pady=(0, 4))

    conf = ttk.Label(info_card, text=confidence_text, style="Small.TLabel")
    conf.grid(row=1, column=0, sticky="w", pady=(0, 12))

    # Naam
    ttk.Label(info_card, text="Naam", style="Section.TLabel").grid(row=2, column=0, sticky="w")
    ttk.Label(info_card, text=name, style="Small.TLabel").grid(row=3, column=0, sticky="w", pady=(2, 10))

    # Beschrijving (scrollbaar)
    ttk.Label(info_card, text="Beschrijving", style="Section.TLabel").grid(row=4, column=0, sticky="w")

    desc_frame = ttk.Frame(info_card)
    desc_frame.grid(row=5, column=0, sticky="nsew", pady=(6, 12))
    desc_frame.columnconfigure(0, weight=1)
    desc_frame.rowconfigure(0, weight=1)

    desc_text = tk.Text(desc_frame, height=7, wrap="word")
    desc_scroll = ttk.Scrollbar(desc_frame, orient="vertical", command=desc_text.yview)
    desc_text.configure(yscrollcommand=desc_scroll.set)

    desc_text.grid(row=0, column=0, sticky="nsew")
    desc_scroll.grid(row=0, column=1, sticky="ns")

    desc_text.insert("1.0", description)
    desc_text.config(state="disabled")

    # Extra info (table JSON) als key/value
    ttk.Label(info_card, text="Extra info", style="Section.TLabel").grid(row=6, column=0, sticky="w")

    meta_frame = ttk.Frame(info_card)
    meta_frame.grid(row=7, column=0, sticky="nsew", pady=(6, 12))
    meta_frame.columnconfigure(0, weight=1)
    meta_frame.rowconfigure(0, weight=1)

    tree = ttk.Treeview(meta_frame, columns=("value",), show="tree headings", height=6)
    tree.heading("#0", text="Eigenschap")
    tree.heading("value", text="Waarde")
    tree.column("#0", width=190, anchor="w")
    tree.column("value", width=620, anchor="w")

    meta_scroll = ttk.Scrollbar(meta_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=meta_scroll.set)

    tree.grid(row=0, column=0, sticky="nsew")
    meta_scroll.grid(row=0, column=1, sticky="ns")

    if isinstance(meta, dict) and meta:
        # beetje nette sortering
        for k in sorted(meta.keys(), key=lambda x: str(x).lower()):
            tree.insert("", "end", text=str(k), values=(str(meta[k]),))
    else:
        tree.insert("", "end", text="(geen)", values=("—",))

    # Buttons
    btn_row = ttk.Frame(info_card)
    btn_row.grid(row=8, column=0, sticky="ew")
    btn_row.columnconfigure(0, weight=1)

    close_btn = ttk.Button(btn_row, text="Sluiten", command=root.destroy)
    close_btn.grid(row=0, column=1, sticky="e")

    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    close_btn.focus_set()
    root.bind("<Return>", lambda event: root.destroy())
    root.bind("<KP_Enter>", lambda event: root.destroy())

    if auto_close_ms is not None:
        root.after(auto_close_ms, root.destroy)

    root.mainloop()


# function to show directory selection window
def show_directory_selection_window():
    """
    shows a GUI window to select a directory.
    args: None
    Returns: str (directory_path) or None if cancelled
    """
    result = {'directory_path': None, 'cancelled': True}
    
    def on_directory_selected():
        # Open directory dialog
        # determine project root: one level above this src file
        try:
            project_root = Path(__file__).resolve().parent.parent
        except Exception:
            project_root = None

        initial_dir = str(project_root) if project_root and project_root.exists() else os.path.expanduser("~")

        dir_path = filedialog.askdirectory(
            title="Selecteer een map",
            initialdir=initial_dir
        )
        
        if dir_path:
            result['directory_path'] = dir_path
            result['cancelled'] = False
            root.quit()
    
    def on_cancel():
        result['cancelled'] = True
        root.quit()
    
    # Create main window
    root = tk.Tk()
    root.title("Mondriaan Detector - Map Selectie")
    root.geometry("500x400")
    root.configure(bg='#f0f0f0')
    root.resizable(False, False)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (250)
    y = (root.winfo_screenheight() // 2) - (200)
    root.geometry(f"500x400+{x}+{y}")
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="30")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Mondriaan Detector", 
                           font=('Arial', 20, 'bold'))
    title_label.grid(row=0, column=0, pady=(0, 10))
    
    # Subtitle
    subtitle_label = ttk.Label(main_frame, text="Selecteer een map", 
                              font=('Arial', 12))
    subtitle_label.grid(row=1, column=0, pady=(0, 30))
    
    # Directory selection option
    directory_frame = ttk.LabelFrame(main_frame, text="Map Selectie", padding="20")
    directory_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 30))
    main_frame.columnconfigure(0, weight=1)
    directory_frame.columnconfigure(0, weight=1)
    
    directory_desc = ttk.Label(directory_frame, text="Selecteer een map\nvan uw computer", 
                              font=('Arial', 10), justify='center')
    directory_desc.grid(row=0, column=0, pady=(0, 15))
    
    directory_button = ttk.Button(directory_frame, text="📁 Selecteer Map", 
                                 command=on_directory_selected,
                                 style='Accent.TButton')
    directory_button.grid(row=1, column=0)
    
    # Cancel button
    cancel_button = ttk.Button(main_frame, text="Annuleren", 
                              command=on_cancel)
    cancel_button.grid(row=3, column=0, pady=(10, 0))
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    
    # Start the GUI
    root.mainloop()
    root.destroy()
    
    if result['cancelled']:
        return None
    else:
        return result['directory_path']
