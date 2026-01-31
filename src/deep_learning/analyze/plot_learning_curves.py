'''
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

this file contains functions to plot training history curves

how to use:
1. plot_learning_curves(history)
    - history: Keras training history object
    - Plots training and validation loss/accuracy over epochs

'''
import matplotlib.pyplot as plt

def plot_learning_curves(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()