import os

import cv2
import numpy as np

import torch
from model import CNN, predict


MODEL_PATH = 'model/cnn.pth'


def main():
    # Load pre-trained model
    if not os.path.exists(MODEL_PATH):
        print("'{}' not found".format(MODEL_PATH))
        return

    model = CNN()
    print(model)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Camera
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()

    while (cap.isOpened()):
        ret, frame = cap.read()

        # Hand ROI
        top_left = (90, 100)
        bottom_right = (top_left[0]+64*3, top_left[1]+64*3)

        # Take ROI crop from frame
        clone = frame.copy()
        roi = clone[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        # Predict
        label = predict(model, roi)

        # Show result right on ROI
        cv2.putText(roi, label, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('ROI', roi)

        # Draw ROI
        cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 1)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
