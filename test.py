import cv2
# Import Classifier for model prediction
from cvzone.ClassificationModule import Classifier
# Import HandDetector for hand localization
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow  # Required for loading Keras model

# --- INITIALIZATION ---
cap = cv2.VideoCapture(1)
# Initialize detector with low confidence to ensure detection [00:41:20]
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Parameters
offset = 20
imgSize = 300
labels = ["A", "B", "C"]  # Should match your Model/labels.txt.

# Classification Setup [00:45:10]
model_path = "Model/keras_model.h5"
labels_path = "Model/labels.txt"
try:
    classifier = Classifier(model_path, labels_path)
except Exception as e:
    print(f"Error initializing Classifier: {e}")
    print(
        "Ensure 'Model' folder exists with 'keras_model.h5' and 'labels.txt', and all libraries (tensorflow) are installed.")
    exit()

while True:
    success, img = cap.read()
    # Create a copy of the original image for drawing the bounding box and text
    imgOutput = img.copy()
    # Find hands, but do not draw the skeleton on the image passed to detector [00:51:00]
    hands, img = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Safely determine crop area to prevent out-of-bounds error
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # --- Resizing Logic (Same as data_collection.py) ---
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # --- Prediction ---
            # Predict the sign using the white-background, 300x300 image
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # --- Drawing on imgOutput ---

            # Draw the main bounding box [00:56:54]
            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)  # Purple Box

            # Draw a filled rectangle for the class label background [00:58:47]
            # Height/Width determined through experimentation in video (approx 90x50)
            cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 90, y1), (255, 0, 255), cv2.FILLED)

            # Put the text label on top of the filled rectangle [00:52:40]
            text_label = labels[index] if index < len(labels) else "UNKNOWN"
            # Text color is white (255, 255, 255)
            cv2.putText(imgOutput, text_label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

            # Optional: Show the intermediate white image for debugging
            cv2.imshow("ImageWhite", imgWhite)

    # Display the final output image with detection and classification
    cv2.imshow("Image Output", imgOutput)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
