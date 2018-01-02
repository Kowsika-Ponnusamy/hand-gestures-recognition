import os
import h5py
import cv2
import numpy as np


DATA_DIR = 'data'


def main():
    # Prepare stored files
    train_file = os.path.join(DATA_DIR, 'train.h5')
    val_file = os.path.join(DATA_DIR, 'val.h5')
    label_file = os.path.join(DATA_DIR, 'labels.txt')

    f_label = open(label_file, 'w')
    f_train = h5py.File(train_file, 'w')
    f_val = h5py.File(val_file, 'w')

    labels = []
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for f in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, f)
        if os.path.isdir(path):
            # get label
            labels.append(f)
            f_label.write('{}\n'.format(f))

            image_paths = [os.path.join(path, image)
                           for image in sorted(os.listdir(path))]
            n = len(image_paths)
            n_train = int(n * 0.8)

            # get 80% for train
            for image_path in image_paths[:n_train]:
                X_train.append(cv2.imread(image_path))
                y_train.append(labels.index(f))

            # 20% for val
            for image_path in image_paths[n_train:]:
                X_val.append(cv2.imread(image_path))
                y_val.append(labels.index(f))

    # Show some infos
    labels = np.array(labels)
    print("Labels {}: {}".format(labels.shape, labels))

    # Cram data into .h5 file
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    f_train.create_dataset('X_train', data=X_train)
    f_train.create_dataset('y_train', data=y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    f_val.create_dataset('X_val', data=X_val)
    f_val.create_dataset('y_val', data=y_val)

    # Close file streams
    f_label.close()
    f_train.close()
    f_val.close()


if __name__ == '__main__':
    main()
