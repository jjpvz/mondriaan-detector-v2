import keras
import matplotlib.pyplot as plt

def augment_images():
    return keras.Sequential([
        keras.layers.RandomShear(x_factor=[0.0, 0.2]),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(height_factor=[-0.2, 0.2]),
        keras.layers.RandomBrightness(0.3),
        keras.layers.RandomContrast(0.3),
    ])

def display_augmented_images(dataset):
    # Pak één batch uit de trainingsset
    for images, labels in dataset.take(1):
        sample_images = images[:5]  # neem 5 voorbeelden
        break

    # Pas augmentatie toe
    augmentation_layer = augment_images()

    augmented = augmentation_layer(sample_images)

    # Plot de originele en geaugmenteerde versie naast elkaar
    plt.figure(figsize=(12, 6))

    for i in range(5):
        # Origineel
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i].numpy().astype("uint8"))
        plt.title("Origineel")
        plt.axis("off")

        # Geaugmenteerd
        plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(augmented[i].numpy().astype("uint8"))
        plt.title("Augmented")
        plt.axis("off")

    plt.show()