"""

"""
import numpy as np
import string
from model.AccuracyPlotter import AccuracyPlotter


class NeuralNetwork:
    def __init__(self, layers, learn_rate: float, epochs: int, accuracy_plotter: AccuracyPlotter):
        self.LAYERS = layers
        self.EPOCHS = epochs
        self.LEARNING_RATE = learn_rate
        self.NUMOUTPUTS = layers[-1].numNodesOut

        keys = string.digits + string.ascii_uppercase + string.ascii_lowercase
        self.outputToIndex = {key: index for index, key in enumerate(keys)}
        self.plotter = accuracy_plotter

    def oneHotEncoding(self, training_labels: np.array):
        oneHotArray = np.zeros((self.NUMOUTPUTS, len(training_labels)))
        for i, label in enumerate(training_labels):
            oneHotArray[self.outputToIndex[str(label)], i] = 1
        return oneHotArray

    def forward_propagation(self, inputs: np.array):
        for layer in self.LAYERS:
            inputs = layer.forward_propagation(inputs)
        return inputs

    def back_propagation(self, outputs: np.array, training_labels: np.array):
        batch_size = len(training_labels)
        one_hot_labels = self.oneHotEncoding(training_labels)
        gradient = outputs - one_hot_labels

        for layer in reversed(self.LAYERS):
            gradient = layer.back_propagation(gradient, batch_size)

    def updateWeightsBiases(self):
        for layer in self.LAYERS:
            layer.updateWeightsBiases(self.LEARNING_RATE)

    def prediction_accuracy(self, outputs: np.array, training_labels: np.array):
        predictions = np.argmax(outputs, axis=0)
        return np.sum(predictions == training_labels) / len(training_labels)

    def train(self, training_images: np.array, training_labels: np.array, batch_size: int, test_images, test_labels):
        num_samples = len(training_labels)
        for epoch in range(self.EPOCHS):
            if epoch % 50 == 0:
                print("Epoch Completed: #" + str(epoch))

            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            training_images = training_images[indices]
            training_labels = training_labels[indices]
            training_accuracy = []

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_images = training_images[start:end]
                batch_labels = training_labels[start:end]
                outputs = self.forward_propagation(batch_images)

                training_accuracy.append(self.prediction_accuracy(outputs, batch_labels))

                self.back_propagation(outputs, batch_labels)
                self.updateWeightsBiases()

            self.plotter.appendTrainingData(epoch=epoch,
                                            accuracy=(sum(training_accuracy) / len(training_accuracy)))
            self.test(test_images, test_labels, epoch)

    def test(self, test_images: np.array, test_labels: np.array, epoch: int):
        outputs = self.forward_propagation(test_images)
        self.plotter.appendTestData(epoch=epoch,
                                    accuracy=self.prediction_accuracy(outputs, test_labels))
