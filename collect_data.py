import argparse
import os

import cv2
import numpy as np


DATA_DIR = 'data'


def main():
    # Create data dir
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True,
                        help='Label for training data')
    parser.add_argument('--n', default=320,
                        help='Number of images collected for each label')
    parser.add_argument('--test', default=False, action='store_true',
                        help='Just test filter, without saving images')
    args = parser.parse_args()

    label_dir = os.path.join(DATA_DIR, args.label)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    # Camera
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()

    i = 0
    while (cap.isOpened() and i < args.n):
        ret, frame = cap.read()

        # Hand ROI
        top_left = (90, 100)
        bottom_right = (top_left[0]+64*3, top_left[1]+64*3)

        # Take ROI crop from frame
        roi = frame[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        # shơw it
        cv2.imshow('ROI', roi)
        # save it
        if not args.test:
            img_path = os.path.join(label_dir, '{}_{}.png'.format(args.label, i))
            cv2.imwrite(img_path, roi)

        # Draw ROI
        cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 1)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
