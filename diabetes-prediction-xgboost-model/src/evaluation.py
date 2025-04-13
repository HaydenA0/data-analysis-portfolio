from sklearn.metrics import accuracy_score, confusion_matrix


def get_confusion_matrix(model, X_val, y_val, threshold=0.5, printit=False):
    val_preds_continuous = model.predict(X_val)

    if val_preds_continuous.ndim > 1 and val_preds_continuous.shape[1] > 1:
        val_preds_continuous = val_preds_continuous[:, 1]
    elif val_preds_continuous.ndim > 1 and val_preds_continuous.shape[1] == 1:
        val_preds_continuous = val_preds_continuous.flatten()

    val_preds_binary = (val_preds_continuous >= threshold).astype(int)

    cm = confusion_matrix(y_val, val_preds_binary)

    if printit:
        print("Confusion Matrix (Validation Set):")
        print(cm)

    return cm


def binarize(Y, treshold=0.7):
    return (Y > treshold).astype(int)


def evaluate_model(model, X_train, X_val, y_train, y_val, threshold=0.5, printit=False):
    train_preds_continuous = model.predict(X_train)
    val_preds_continuous = model.predict(X_val)

    train_preds_binary = (train_preds_continuous >= threshold).astype(int)
    val_preds_binary = (val_preds_continuous >= threshold).astype(int)

    train_acc = accuracy_score(y_train, train_preds_binary)
    val_acc = accuracy_score(y_val, val_preds_binary)

    if printit:
        print(f"Accuracy on training data: {round(100 * train_acc, 2)}%")
        print(f"Accuracy on validation data: {round(100 * val_acc, 2)}%")

    return train_acc, val_acc
