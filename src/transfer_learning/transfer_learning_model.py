from tensorflow import keras
from tensorflow.keras import layers

def transfer_learning_model(input_shape, num_classes, augmentation=None,
                            dense_units=256, dropout=0.3, lr=1e-4):
    inputs = keras.Input(shape=input_shape)

    x = inputs
    if augmentation is not None:
        x = augmentation(x)

    # MobileNetV2 verwacht preprocess_input op raw RGB (0..255)
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet"
    )

    # eerst volledig bevriezen (belangrijk!)
    base_model.trainable = False
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def finetune_mobilenetv2(model, fine_tune_at=120, lr=1e-5):
    """
    Unfreeze vanaf layer fine_tune_at en train met kleine lr.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.lower().startswith("mobilenetv2"):
            base_model = layer
            break

    # fallback: vaak is base_model de 3e/4e layer in graph; maar bovenstaande is netter
    if base_model is None:
        # zoek eerste nested keras.Model
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                base_model = layer
                break

    if base_model is None:
        raise ValueError("Base model (MobileNetV2) niet gevonden in het model.")

    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
