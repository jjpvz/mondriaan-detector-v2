from dataAcquisition.load import load_model
from helpers.resize import resize_image
from preprocessing.pipeline import preprocess_image
from featureExtraction.pipeline import extract_features
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from segmentation.colorSegmentation import segment_colors


def select_image_gui():
    """
    Opent een grafische interface om een afbeelding te selecteren.
    Retourneert de afbeelding als OpenCV numpy array (BGR formaat).
    """
    # Maak een root window (verborgen)
    root = tk.Tk()
    root.withdraw()  # Verberg het hoofdvenster
    root.attributes('-topmost', True)  # Zet dialoog op voorgrond
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Selecteer een afbeelding",
        filetypes=[
            ("Afbeeldingen", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Alle bestanden", "*.*")
        ]
    )
    
    # Sluit de root window
    root.destroy()
    
    # Laad de afbeelding met OpenCV als er een bestand is geselecteerd
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Fout: Kon afbeelding niet laden: {file_path}")
            return None, None
        print(f"Afbeelding geselecteerd: {file_path}")
        print(f"Afbeelding dimensies: {img.shape}")
        return img, file_path
    else:
        print("Geen afbeelding geselecteerd")
        return None, None


def show_prediction_gui(img, predicted_class, confidence, file_path, continue_callback):
    """
    Toont een GUI venster met de geselecteerde afbeelding en de voorspelde klasse.
    
    Parameters:
    - img: OpenCV afbeelding (BGR formaat)
    - predicted_class: De voorspelde klasse (string of int)
    - confidence: Confidence score in procenten (float)
    - file_path: Pad naar het originele bestand
    - continue_callback: Functie die wordt aangeroepen voor een nieuwe voorspelling
    """
    # Maak hoofdvenster
    root = tk.Tk()
    root.title("Mondriaan Detector - Voorspelling")
    root.geometry("700x680")
    root.configure(bg='#f0f0f0')
    root.resizable(False, False)
    
    # Centreer venster op scherm
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Zet venster op de voorgrond
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.focus_force()
    
    # Titel label
    title_label = tk.Label(
        root, 
        text="Classificatie Resultaat",
        font=("Arial", 16, "bold"),
        bg='#f0f0f0',
        fg='#333333'
    )
    title_label.pack(pady=8)
    
    # Converteer OpenCV (BGR) naar PIL (RGB) voor weergave
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Resize afbeelding voor weergave (behoud aspect ratio)
    max_size = (450, 300)
    pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Converteer naar PhotoImage
    photo = ImageTk.PhotoImage(pil_img)
    
    # Frame voor afbeelding
    img_frame = tk.Frame(root, bg='#ffffff', relief=tk.SUNKEN, borderwidth=2)
    img_frame.pack(pady=8, padx=20)
    
    # Toon afbeelding
    img_label = tk.Label(img_frame, image=photo, bg='#ffffff')
    img_label.image = photo  # Bewaar referentie
    img_label.pack(padx=6, pady=6)
    
    # Bestandsnaam label
    filename = file_path.split('/')[-1].split('\\')[-1]
    file_label = tk.Label(
        root,
        text=f"Bestand: {filename}",
        font=("Arial", 9),
        bg='#f0f0f0',
        fg='#666666'
    )
    file_label.pack(pady=6)
    
    # Resultaat frame
    result_frame = tk.Frame(root, bg='#e8f4f8', relief=tk.RAISED, borderwidth=3)
    result_frame.pack(pady=10, padx=35, fill=tk.X)
    
    # Voorspelling label
    prediction_label = tk.Label(
        result_frame,
        text="Voorspelde Klasse:",
        font=("Arial", 12, "bold"),
        bg='#e8f4f8',
        fg='#333333'
    )
    prediction_label.pack(pady=10)
    
    # Klasse waarde met kleur
    class_color = '#2e7d32' if 'M0' in str(predicted_class) or 'mondriaan' in str(predicted_class).lower() else '#d32f2f'
    class_value_label = tk.Label(
        result_frame,
        text=str(predicted_class),
        font=("Arial", 22, "bold"),
        bg='#e8f4f8',
        fg=class_color
    )
    class_value_label.pack(pady=6)
    
    # Confidence score label
    confidence_label = tk.Label(
        result_frame,
        text="Vertrouwen:",
        font=("Arial", 11, "bold"),
        bg='#e8f4f8',
        fg='#333333'
    )
    confidence_label.pack(pady=(12, 4))
    
    # Confidence waarde met kleurcodering
    if confidence >= 80:
        confidence_color = '#2e7d32'  # Groen voor hoge confidence
    elif confidence >= 60:
        confidence_color = '#f57c00'  # Oranje voor middelmatige confidence
    else:
        confidence_color = '#d32f2f'  # Rood voor lage confidence
    
    confidence_value_label = tk.Label(
        result_frame,
        text=f"{confidence:.2f}%",
        font=("Arial", 18, "bold"),
        bg='#e8f4f8',
        fg=confidence_color
    )
    confidence_value_label.pack(pady=(4, 12))
    
    # Button frame voor knoppen naast elkaar
    button_frame = tk.Frame(root, bg='#f0f0f0')
    button_frame.pack(pady=15)
    
    # Nieuwe voorspelling knop
    new_prediction_button = tk.Button(
        button_frame,
        text="Nieuwe Voorspelling",
        font=("Arial", 10, "bold"),
        bg='#2e7d32',
        fg='white',
        activebackground='#1b5e20',
        activeforeground='white',
        command=lambda: [root.destroy(), continue_callback()],
        padx=18,
        pady=8,
        cursor='hand2',
        relief=tk.RAISED,
        borderwidth=2
    )
    new_prediction_button.pack(side=tk.LEFT, padx=8)
    
    # Sluit knop
    close_button = tk.Button(
        button_frame,
        text="Afsluiten",
        font=("Arial", 10, "bold"),
        bg='#d32f2f',
        fg='white',
        activebackground='#b71c1c',
        activeforeground='white',
        command=root.destroy,
        padx=25,
        pady=8,
        cursor='hand2',
        relief=tk.RAISED,
        borderwidth=2
    )
    close_button.pack(side=tk.LEFT, padx=8)
    
    # Start GUI
    root.mainloop()


def process_single_prediction(rf_model):
    """
    Verwerkt één enkele voorspelling.
    
    Parameters:
    - rf_model: Het getrainde Random Forest model
    """
    img, file_path = select_image_gui()
    
    if img is None:
        print("Geen geldige afbeelding geselecteerd.")
        return False
    
    print("Afbeelding verwerken...")
    img_resized = resize_image(img, 1920, 1080)

    pre_img = preprocess_image(img_resized, False)

    red_mask, yellow_mask, blue_mask = segment_colors(pre_img, False)

    print("Features extraheren...")
    features = extract_features(0, "input_image", "unknown", pre_img, red_mask, yellow_mask, blue_mask)
    feature_df = pd.DataFrame([features])

    X = feature_df.drop(columns=["id", "filename", "class"])

    # Zorg ervoor dat alle verwachte features aanwezig zijn
    expected_features = rf_model.feature_names_in_
    
    # Voeg ontbrekende features toe met waarde 0
    for feature in expected_features:
        if feature not in X.columns:
            X[feature] = 0
    
    # Zorg voor juiste volgorde van features
    X = X[expected_features]

    print("Voorspelling maken...")
    prediction = rf_model.predict(X)
    predicted_class = prediction[0]
    prediction_proba = rf_model.predict_proba(X)
    confidence = np.max(prediction_proba) * 100
    print(f"Voorspelde klasse: {predicted_class} met een vertrouwen van {confidence:.2f}%")
    
    # Toon resultaat in GUI met callback voor nieuwe voorspelling
    show_prediction_gui(img, predicted_class, confidence, file_path, 
                       lambda: process_single_prediction(rf_model))
    
    return True


def random_forest_predict():
    """
    Start de voorspelling loop. Laadt het model en blijft voorspellingen doen
    totdat de gebruiker op 'Afsluiten' drukt.
    """
    # Laad het getrainde model (slechts één keer)
    print("Model laden...")
    rf_model = load_model("RF_model")
    print("Model geladen! Klaar voor voorspellingen.\n")
    
    # Start eerste voorspelling
    process_single_prediction(rf_model)





