# python train.py --az handwritten_data.csv --model ocrmodel/ocr.keras

import matplotlib
matplotlib.use("Agg")
from neuralnet.models import OCRModel
from neuralnet.dataset import load_mnist_data, load_og_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler  
from imutils import build_montages
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import argparse
import cv2 
    
def load_input_data(data_path):
    Data, Labels = load_og_data(data_path)
    digitsData, digitsLabels = load_mnist_data()
    Labels += 10
    data = np.vstack([Data, digitsData])
    labels = np.hstack([Labels, digitsLabels])
    data = np.array([cv2.resize(image, (32, 32)) for image in data], dtype="float32")
    data = np.expand_dims(data, axis=-1)
    data /= 255.0
    return data, labels

def split_data(data, labels):
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
    return trainX, testX, trainY, testY

def augment_data(trainX, trainY):
    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        fill_mode="nearest")
    return aug.flow(trainX, trainY, batch_size=BATCH_SIZE)

def compile_model(input_shape, num_classes, kernel_sizes, filters, reg, learning_rate):
    opt = SGD(learning_rate=learning_rate)
    model = OCRModel.build_model(input_shape[0], input_shape[1], input_shape[2], num_classes, kernel_sizes, filters, reg)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

def train_model(model, train_data, test_data, steps_per_epoch, epochs, class_weight, model_checkpoint, lr_scheduler):
    H = model.fit(train_data, validation_data=test_data, steps_per_epoch=steps_per_epoch, epochs=epochs, class_weight=class_weight, callbacks=[model_checkpoint, lr_scheduler], verbose=1)
    return H

def model_eval(model, testX, testY, labelNames):
    predictions = model.predict(testX, batch_size=BATCH_SIZE)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

def save_trained_model(model, model_path):
    model.save(model_path, save_format="h5")

def plot_training_history(history, plot_path):
    N = np.arange(0, len(history.history["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)

def visualize_results(model, testX, testY, labelNames):
    images = []
    for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
        probs = model.predict(testX[np.newaxis, i])
        prediction = probs.argmax(axis=1)
        label = labelNames[prediction[0]]
        image = (testX[i] * 255).astype("uint8")
        color = (0, 255, 0) if prediction[0] == np.argmax(testY[i]) else (0, 0, 255)
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        images.append(image)
    montage = build_montages(images, (96, 96), (7, 7))[0]
    cv2.imshow("OCR Results", montage)
    cv2.waitKey(0)

def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--az", required=True, help="path to A-Z dataset")
    ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained ocr recognition model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output training history file")
    args = vars(ap.parse_args())

    EPOCHS = 50
    BATCH_SIZE = 128

    print("[INFO] loading datasets...")
    data, labels = load_input_data(args["az"])
    le = LabelBinarizer()
    labels = le.fit_transform(labels)
    classTotals = labels.sum(axis=0)
    classWeight = {i: classTotals.max() / classTotals[i] for i in range(len(classTotals))}
    trainX, testX, trainY, testY = split_data(data, labels)
    train_data = augment_data(trainX, trainY)
    print("[INFO] compiling model...")
    lr = 1e-1
    model = compile_model((32, 32, 1), len(le.classes_), (3, 3, 3), (64, 64, 128, 256), 0.0005, lr)
    model_checkpoint = ModelCheckpoint(filepath=args["model"], monitor='val_loss', save_best_only=True, save_weights_only=False)
    save_trained_model(model, "ocr.keras")
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: scheduler(epoch, lr))

    print("[INFO] training network...")
    history = train_model(model, train_data, (testX, testY), len(trainX) // BATCH_SIZE, EPOCHS, classWeight, model_checkpoint, lr_scheduler)
    print("[INFO] Saving trained model...")
    print("[INFO] Model saved !!!!!!!")

    labelNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    model_eval(model, testX, testY, labelNames)
    plot_training_history(history, args["plot"])
    visualize_results(model, testX, testY, labelNames)
