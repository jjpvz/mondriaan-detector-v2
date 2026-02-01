# Mondriaan Detector v2

Dit project bevat drie verschillende benaderingen voor het classificeren van Mondriaan-stijlen: Machine Learning (Random Forest), Deep Learning (Custom CNN), en Transfer Learning (MobileNetV2)

## 1. Setup & installatie

1\. Virtual Environment opzetten:

```
python -m venv venv
.\venv\Scripts\Activate.ps1 

Gebruik .ps1 voor powershell
Gebruik .bat voor command prompt
Laat leeg voor bash/linux/mac
```

2\. Dependencies installeren:

```
pip install -r requirements.txt
```

3\. Configuratie instellen: Maak een config.ini aan in de project root.

Create a config.ini file in the project root:

```
# config.ini
[General]
fullset_path = data/dataset
subset_path = data/subset
csv_path = temp/dataframe
texts_csv_path = testing/texts.csv
ml_path = models/mondriaan_detector_ml.joblib
dl_path = models/mondriaan_detector_dl.keras
tl_path = models/mondriaan_detector_tl.keras
```

## 2. Project flow & gebruik

Het project is opgesplitst in twee fases: trainen en testen.

### Fase A: Modellen trainen (train.py)

Gebruik dit script om de modellen te genereren. Je kunt kiezen welk specifiek model je wilt trainen.

Train machine learning model:

```
python src/train.py ml
```

Train deep learning model:

```
python src/train.py dl
```

Train transfer learning model:

```
python src/train.py tl
```

### Fase B: Modellen testen met GUI (test.py)

Gebruik dit script om de gegeneerde modellen te testen. Je kunt kiezen welk specifiek model je wilt testen.

Train machine learning model:

```
python src/test.py ml
```

Train deep learning model:

```
python src/test.py dl
```

Train transfer learning model:

```
python src/test.py tl
```

### 3. Systeem configuratie

#### 1. Software Requirements

- Windows 11
- Python 3.11.9
- Vereiste Python-pakketten (zie requirements.txt)

#### 2. Hardware Requirements

- Minimaal 8 GB RAM
- Minimaal 4-core CPU
- Opslagruimte: Minimaal 2 GB

## 4. Dataset acquisitie

Er zijn twee manieren om de benodigde data te verkrijgen:

### Optie A: Geautoriseerde toegang

- Locatie: De data is opgeslagen in een beveiligde OneDrive-omgeving
  [OneDrive Dataset Link](https://hannl-my.sharepoint.com/my?id=%2Fpersonal%2Fjjp%5Fvanzwol%5Fstudent%5Fhan%5Fnl%2FDocuments%2FEVML%2FFoto&viewid=2a1c220c%2D5c35%2D4559%2Da5da%2D77e30261edc9).
- Installatie na autorisatie:
  1.  Download het gedeelde ZIP-bestand.
  2.  Pak de inhoud uit in de data/ folder van de projectroot.
  3.  Controleer of de paden in config.ini overeenkomen met je lokale opslaglocatie.

### Optie B: Zelfstandige acquisitie

Je kunt zelf een dataset samenstellen om de robuustheid van de code te testen.

- Instructies: Volg hiervoor de gedetailleerde stappen en criteria zoals beschreven in de onderzoeksverslagen. Deze documenten zijn hier te vinden: [Machine learning onderzoeksverslag](docs/ml_verslag.pdf) & [Deep learning onderzoeksverslag](docs/dl_verslag.pdf).
