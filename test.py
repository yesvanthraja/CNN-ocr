# python test.py --model ocrmodel/ocr.keras --image images/hello_world.png

import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from imutils.contours import sort_contours


def extract_sentence_from_image(image_path, model_path):
    print("[INFO] loading handwriting OCR model...")
    model = tf.keras.models.load_model(model_path)
    image = cv2.imread(image_path)
    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="top-to-bottom")[0]
    chars = []

    prev_y = None
    line = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            else:
                thresh = imutils.resize(thresh, height=32)
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            chars.append((padded, (x, y, w, h)))

            if prev_y is None or abs(y - prev_y) < h:
                line.append((padded, (x, y, w, h)))
            else:
                chars.append(line)
                line = [(padded, (x, y, w, h))]

            prev_y = y
    
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]

    if line:
        chars.append(line)

    extracted_text = []

    for line in chars:
        if isinstance(line, list): 
            line_text = ""
            prev_x = None

            for (pred, (x, y, w, h)) in sorted(line, key=lambda x: x[1][0]):
                pred = model.predict(np.expand_dims(pred, axis=0))[0]
                i = np.argmax(pred)
                prob = pred[i]
                label = labelNames[i]
                print("[INFO] {} - {:.2f}%".format(label, prob * 100))
                if prev_x is not None and x - prev_x > w:
                    line_text += " "  

                line_text += label  
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                prev_x = x
            extracted_text.append(line_text)

    return extracted_text, output_image

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to trained handwriting recognition model")
    args = vars(ap.parse_args())

    extracted_sentence, output_image = extract_sentence_from_image(args["image"], args["model"])
    print("Extracted Sentence:\n", extracted_sentence)
    cv2.imshow("OCR Results", output_image)
    cv2.waitKey(0)