from tensorflow import keras
from tensorflow.keras import layers


def create_transfer_model_efficientnet(num_classes, input_shape=(224, 224, 3)):
    """
    Transfer learning met EfficientNetB0.

    Belangrijk:
    - prepare_data_for_dl geeft al beelden in het bereik [0, 1]
    - EfficientNetB0 verwacht ruwe beelden (0–255) + eigen preprocess_input
    - Daarom schalen we hier eerst terug naar 0–255, dán preprocess_input
    """

    # 1. EfficientNetB0 backbone (ImageNet gewichten, zonder top)
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # backbone bevriezen → alleen onze eigen lagen worden getraind
    base_model.trainable = False

    # 2. Input-layer
    inputs = keras.Input(shape=input_shape)

    # We krijgen al [0,1] van prepareData.py → eerst terug naar [0,255]
    x = layers.Lambda(lambda im: im * 255.0)(inputs)

    # Officiële EfficientNet-preprocessing toepassen
    x = keras.applications.efficientnet.preprocess_input(x)

    # 3. Features uit de backbone
    x = base_model(x, training=False)

    # 4. Eigen classificatielaag erboven
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="efficientnetb0_transfer")
    return model
