# Mondriaan Detector v2

## Getting Started

Follow these steps to set up and run the project.

1\. Create a virtual environment

```
python -m ven venv
```

2\. Activate the virtual environment (from project root)

Windows (PowerShell):

```
venv\Scripts\Activate.ps1
```

3\. Install dependencies

```
pip install -r requirements.txt
```

4\. Create the configuration file

Create a config.ini file in the project root:

```
# config.ini
[General]
subset_path = data/subset
fullset_path = data/fullset
```

- `subset_path` points to a small subset of images for testing.
- `fullset_path` points to the full dataset.

5\. Prepare your dataset

```
data/
├─ fullset/
│  ├─ M01/
│  │  ├─ M1 (1).JPG
│  │  ├─ M1 (2).JPG
│  │  └─ ...
│  ├─ M02/
│  │  └─ ...
│  └─ ...
└─ subset/
   ├─ subset/
   │  ├─ 1 (1).JPG
   │  ├─ mondriaan1 (1).JPG
   │  └─ ...
```

6\. Run the main script (from project root)

```
python src/main.py
```
