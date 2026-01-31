import keras
from deepLearning import get_predictions
import keras_tuner as kt


def find_optimal_model(input_shape, num_classes, train_ds, val_ds, data_augmentation_layer):
    my_builder = get_tuner_model(input_shape, num_classes, data_augmentation_layer)

    tuner = kt.Hyperband(my_builder, 
                         objective='val_accuracy', 
                         max_epochs=40, 
                         factor = 3,
                         directory = 'tuner_example',
                         project_name = 'intro_to_kt')
    
    tuner.search(train_ds, validation_data=val_ds)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    tuner.search_space_summary()

    best_model = tuner.get_best_models()[0]
    best_model.summary()
    best_model.evaluate(val_ds)

    y_val, y_pred = get_predictions(best_model, val_ds)

def get_tuner_model(input_shape, num_classes, data_augmentation_layer):

    def model_builder(hp):
        model = keras.Sequential()
        model.add(keras.Input(shape=input_shape))

        model.add(data_augmentation_layer)
        model.add(keras.layers.Rescaling(1./255))
        model.add(keras.layers.Normalization(axis=-1))
        
        hp_filters_1 = hp.Int('filters_1', min_value=8, max_value=32, step=8)
        model.add(keras.layers.Conv2D(hp_filters_1, (7, 7), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        
        hp_filters_2 = hp.Int('filters_2', min_value=16, max_value=64, step=16)
        model.add(keras.layers.Conv2D(hp_filters_2, (7, 7), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        hp_filters_3 = hp.Int('filters_3', min_value=32, max_value=128, step=32)
        model.add(keras.layers.Conv2D(hp_filters_3, (7, 7), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(rate=hp_dropout))
        
        model.add(keras.layers.Flatten())

        hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(keras.layers.Dense(hp_units, activation='relu'))

        model.add(keras.layers.Dense(num_classes, activation='softmax'))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model
    
    return model_builder