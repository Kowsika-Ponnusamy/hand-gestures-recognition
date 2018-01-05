# Hand gestures recognition

This repository is mainly for CNN research, on theme of hand gestures classification.

![demo](demo.png)

## Coursera's SIGNS dataset

This dataset is provided in Course 4 of [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning).
Check out:
+ `data/train_signs.h5`: Train dataset with 1080 images (64x64 RGB) with 6 labels (from `0` to `5`)
+ `data/test_signs.h5`: Validation dataset, 120 images, same characteristics
+ [coursera_cnn.ipynb](https://nbviewer.jupyter.org/github/hiepph/hand-gestures-recognition/blob/master/coursera_cnn.ipynb): Notebooks for using CNN to classify this dataset, 92% validation accuracy

## Handmade dataset

If you want to collect your own data and test model in real-time prediction, do following steps:

+ Collect data: `python collect_data.py --label 0 --n 300` for collecting 300 images for label `0`.
Feel free to collect as many labels as you want, with `--n` flag big enough. And put your hand in Region of Interest (green square).
Collected data will live in `data` folder with `data/labels.txt` contains all collected labels.

+ Pack data: `python pack_data.py` for packing all collected data into `data/train.h5` and `data/val.h5`

+ Build and train model: open [cnn.ipynb](https://nbviewer.jupyter.org/github/hiepph/hand-gestures-recognition/blob/master/cnn.ipynb) and follow guide on CNN research. Trained model's weights will be stored in `model/cnn.pth`

+ Real-time prediction: `python main.py`, it will auto load trained model weights and predict your hand gestures in Region of Interest.

I've already included trained model's weights (`model/cnn.pth`) with 6 labels (from `0` to `5`). You can instantly run demo test with `python main.py`
