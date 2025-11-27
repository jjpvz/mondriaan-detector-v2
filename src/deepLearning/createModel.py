from tensorflow import keras

def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        # 1. CONVOLUTIONAL LAAG: Leert features te extraheren
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        
        # 2. Extra CONVOLUTIONAL LAAG
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # 3. De output "platte" maken (Flatten)
        keras.layers.Flatten(),
        
        # 4. DENSE LAAG (Klassieke Neurale Netwerk): Voor classificatie
        keras.layers.Dense(128, activation='relu'),
        
        # 5. OUTPUT LAAG: Aantal eenheden gelijk aan het aantal klassen
        keras.layers.Dense(num_classes, activation='softmax') # 'softmax' voor multi-klasse classificatie
    ])
    return model