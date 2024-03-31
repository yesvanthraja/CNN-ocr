from tensorflow.keras.datasets import mnist
import numpy as np

def load_og_data(data_path):
    data = []
    labels = []
    for i in open(data_path):
        row = i.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype = "uint8")
        image = image.reshape((28, 28))
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")
    return (data, labels)




def load_mnist_data():
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	data = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])
	return (data, labels)
