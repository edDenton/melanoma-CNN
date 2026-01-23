"""

"""

import numpy as np
import pandas as pd
import cv2

from layers import NeuralNetwork, Conv2D, Flatten, DenseLayer
from AccuracyPlotter import AccuracyPlotter

FILEPATH = "melanomaData/"


def process_image(image_path):
    image = cv2.imread(image_path)
    resizedImage = cv2.resize(image, (128, 128))
    resizedImage = resizedImage / 255.0
    resizedImage = np.transpose(resizedImage, (2, 0, 1))
    return resizedImage


def build_dataset():
    dataset = pd.read_csv(FILEPATH + "train.csv")

    benign_rows = dataset[dataset['benign_malignant'] == 'benign']
    malignant_rows = dataset[dataset['benign_malignant'] == 'malignant']

    benign_rows = benign_rows.sample(n=1000)

    combined_dataset = pd.concat([malignant_rows, benign_rows])
    combined_dataset = combined_dataset.sample(frac=1).reset_index(drop=True)

    benign_images = []
    malignant_images = []
    benign_label = []
    malignant_label = []

    for row in combined_dataset.itertuples():
        filename = row[1]
        label = row[7]

        image_path = FILEPATH + "train/" + filename + ".jpg"
        image_tensor = process_image(image_path)

        if label == "benign":
            benign_images.append(image_tensor)
            benign_label.append(0)
        else:
            malignant_images.append(image_tensor)
            malignant_label.append(1)

    benign_image_array = np.array(benign_images)
    malignant_image_array = np.array(malignant_images)
    benign_label = np.array(benign_label)
    malignant_label = np.array(malignant_label)

    return benign_image_array, malignant_image_array, benign_label, malignant_label


def getData(b_images, m_images, b_labels, m_labels):
    b_perm = np.random.permutation(len(b_images))
    m_perm = np.random.permutation(len(m_images))

    b_images, b_labels = b_images[b_perm], b_labels[b_perm]
    m_images, m_labels = m_images[m_perm], m_labels[m_perm]

    X_train = np.concatenate((b_images[:400], m_images[:400]), axis=0)
    y_train = np.concatenate((b_labels[:400], m_labels[:400]), axis=0)

    X_test = np.concatenate((b_images[400:], m_images[400:]), axis=0)
    y_test = np.concatenate((b_labels[400:], m_labels[400:]), axis=0)

    train_perm = np.random.permutation(len(y_train))
    test_perm = np.random.permutation(len(y_test))

    X_train, y_train = X_train[train_perm], y_train[train_perm]
    X_test, y_test = X_test[test_perm], y_test[test_perm]

    return X_train, y_train, X_test, y_test


def main():
    LR = 0.005
    EPOCHS = 350
    LAYERS = [Conv2D(3, 16, 3, 1, 1),
              Conv2D(16, 32, 3, 2, 1),
              Flatten(),
              DenseLayer(131072, 128, False),
              DenseLayer(128, 2, True)]
    BATCH_SIZE = 256

    dataPlotter = AccuracyPlotter(learn_rate=LR, epochs=EPOCHS, layers=LAYERS, batch_size=BATCH_SIZE)
    neural_network = NeuralNetwork(layers=LAYERS, learn_rate=LR, epochs=EPOCHS, accuracy_plotter=dataPlotter)

    benign_images, malignant_images, benign_labels, malignant_labels = build_dataset()
    training_images, training_labels, testing_images, testing_labels = getData(benign_images, malignant_images, benign_labels, malignant_labels)

    print("Data has been gathered")
    neural_network.train(training_images, training_labels, BATCH_SIZE)
    print("Finished Training")
    neural_network.test(testing_images, testing_labels)
    print("Finished Testing")
    dataPlotter.showPlot()
    # seePerformance(neural_network, training_images, training_labels)


if __name__ == '__main__':
    main()