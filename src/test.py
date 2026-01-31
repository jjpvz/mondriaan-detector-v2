import argparse
from test_cnn_model import test_cnn_model_with_gui
from machine_learning.testing.test_model import test_random_forest_with_gui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GUI testen voor ML en DL modellen.")
    parser.add_argument(
        'model', 
        choices=['ml', 'dl', 'ts'], 
        help="Welk model wil je testen? 'ml' voor Random Forest, 'dl' voor CNN, of 'tl' voor MobileNetV2."
    )

    args = parser.parse_args()

    if args.model == 'ml':
        test_random_forest_with_gui()

    if args.model == 'dl' or args.model == 'ts':
        test_cnn_model_with_gui()
