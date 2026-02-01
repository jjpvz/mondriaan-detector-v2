import numpy as np

def get_predictions(model, val_ds):
    X_val = []
    y_val = []

    for batch_imgs, batch_labels in val_ds:
        X_val.append(batch_imgs.numpy())
        y_val.append(batch_labels.numpy())

    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    pred_probs = model.predict(X_val)
    y_pred = np.argmax(pred_probs, axis=1)

    return y_val, y_pred