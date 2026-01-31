import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_transfer_model(num_classes, input_shape=(224, 224, 3)):
    """
    CNN met transfer learning op basis van MobileNetV2.
    - Onderste lagen (base_model) zijn voorgetraind op ImageNet
    - Bovenste lagen (classifier) zijn nieuw voor ons klassen
    """

    # 1. Basisnetwerk laden (pretrained op ImageNet)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,       # geen originele classifier
        weights="imagenet",      # gebruik gewichten van anderen
    )

    # 2. Basis "freezen": deze lagen trainen we in eerste instantie niet mee
    base_model.trainable = False

    # 3. Eigen classifier erboven bouwen
    inputs = keras.Input(shape=input_shape)

    # Preprocessing die hoort bij MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Feature extractor
    x = base_model(x, training=False)

    # Van feature maps naar vector
    x = layers.GlobalAveragePooling2D()(x)

    # Kleine dense-laag
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)  # beetje regularisatie

    # Outputlaag: aantal units = aantal klassen
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mondriaan_transfer_model")

    # 4. Model compileren (learning rate iets lager dan standaard)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    return model