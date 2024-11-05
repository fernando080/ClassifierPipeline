import os
import tempfile
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
N_CHANNELS = 3
N_CLASSES = 2
BATCH_SIZE = 600
BUFFER_SIZE = 1000
SEED = 42
IMG_HEIGHT = 512
IMG_WIDTH = 640
EPOCHS = 400
AUTOTUNE = tf.data.experimental.AUTOTUNE
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR')
]


def load_and_preprocess_data(file_path: str):
    """Load and preprocess dataset."""
    data = pd.read_excel(file_path, index_col=0)
    data = data.iloc[:, 3:-1].fillna(0)

    # Display class distribution
    neg, pos = np.bincount(data['target'])
    total = neg + pos
    print(f"Examples:\n    Total: {total}\n    Positive: {pos} ({100 * pos / total:.2f}% of total)\n")

    # Split features and labels
    labels = np.array(data.pop('target'))
    features = np.array(data)

    return features, labels


def scale_features(train_features, val_features):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # Clipping values for stability
    train_features = np.clip(train_features, -5, 5)
    val_features = np.clip(val_features, -5, 5)

    return train_features, val_features


def define_model(train_features, metrics=METRICS, output_bias=None):
    """Define and compile a neural network model."""
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = keras.Sequential([
        keras.layers.Dense(20, activation='relu', input_shape=(train_features.shape[-1],)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-3),
                  loss='binary_crossentropy',
                  metrics=metrics)
    return model


def plot_confusion_matrix(labels, predictions, p=0.5):
    """Plot confusion matrix for model predictions."""
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'Confusion Matrix @ {p:.2f}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total Positives: ', np.sum(cm[1]))


def setup_gpu():
    """Enable GPU memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)


def get_callbacks():
    """Define callbacks for model training."""
    logdir = os.path.join("./models/nn/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    return [
        tensorboard_callback,
        keras.callbacks.EarlyStopping(patience=19, monitor='val_prc', mode='max', restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('./models/risk_nn_model.h5', monitor='val_prc', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_prc', factor=0.1, patience=11, min_lr=1e-7)
    ]


def calculate_class_weights(labels):
    """Calculate class weights to handle class imbalance."""
    neg, pos = np.bincount(labels)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    return {0: weight_for_0, 1: weight_for_1 * 10}


def main():
    # Load and preprocess data
    train_features, train_labels = load_and_preprocess_data('./dataset/data_test.xlsx')
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=SEED)

    # Scale features
    train_features, val_features = scale_features(train_features, val_features)

    # Set up GPU
    setup_gpu()

    # Initialize model with output bias
    initial_bias = np.log([np.bincount(train_labels)[1] / np.bincount(train_labels)[0]])
    model = define_model(train_features, output_bias=initial_bias)

    # Callbacks and class weights
    callbacks = get_callbacks()
    class_weight = calculate_class_weights(train_labels)

    # Train the model
    history = model.fit(
        train_features, train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_features, val_labels),
        callbacks=callbacks,
        class_weight=class_weight
    )

    # Evaluate model and plot confusion matrix
    val_predictions = model.predict(val_features, batch_size=BATCH_SIZE)
    plot_confusion_matrix(val_labels, val_predictions)


if __name__ == "__main__":
    main()