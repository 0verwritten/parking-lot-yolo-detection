# Parking Lot YOLO Detection Project

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The Parking Lot YOLO Detection Project is an object detection system based on the YOLO (You Only Look Once) algorithm. This project aims to detect vehicles in parking lots and provide real-time analysis to assist with parking management, security, and traffic optimization.

## Features
- Real-time vehicle detection in parking lot images or video streams.
- Pre-trained YOLO model for easy setup and quick deployment.
- Ability to train the YOLO model on custom datasets.
- Efficient and fast object detection.

## Installation
1. Clone the repository:

```
git clone https://github.com/0verwritten/parking-lot-yolo-detection.git
cd parking-lot-yolo-detection
```

2. Set up the virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate   # For Windows, use: venv\Scripts\activate
```

3. Install the required dependencies (!TODO!):

```
pip install -r requirements.txt
```

## Usage
Everything is located in [main jupiter file](./main.ipynb)
<!-- To use the pre-trained YOLO model for parking lot vehicle detection, follow these steps:

1. Download the pre-trained YOLO weights from [here](link_to_pretrained_weights) and place them in the `weights` folder.

2. Use the following command to run the detection on an image (!TODO!):

```
python detect_image.py --image path/to/your/image.jpg
```

3. To perform real-time vehicle detection on a video stream, use (!TODO!):

```
python detect_video.py --video path/to/your/video.mp4
``` -->

## Dataset
For training the YOLO model on custom parking lot datasets, you need to organize your data in the following format:

```
dataset/
|-- images/
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
|
|-- labels/
|   |-- image1.txt
|   |-- image2.txt
|   |-- ...
|
|-- classes.txt
```

- The `images` folder should contain all the images in JPG or PNG format.
- The `labels` folder should contain the corresponding label files in YOLO format. Each label file should have one row per object detected in the image in the following format: `<class_id> <x_center> <y_center> <width> <height>`, where `(x_center, y_center)` represents the center coordinates of the object, and `(width, height)` represent its dimensions, all relative to the image width and height.
- The `classes.txt` file should list all the class names, one per line.

## Training
This script enables you to train a YOLO (You Only Look Once) object detection model using the Ultralytics YOLO framework. The script supports both local datasets and datasets hosted on Roboflow. It provides various training configuration options and supports exporting the trained model in different formats.
#### To train the YOLO model on your custom dataset, follow these steps:

1. Organize your dataset as described in the [Dataset](#dataset) section.

2. Modify the `./dataset/Parking.../config.yaml` file to set the training parameters, such as batch size, learning rate, etc.

3. Start the training process using the following command:

```
python train.py --dataset path/to/your/dataset/ --config config/config.yaml
```

### Arguments

- `--roboflow-project`: Name of your Roboflow project.
- `--roboflow-project-version`: Version of your Roboflow project (default: 1).
- `--roboflow-api-key`: Your Roboflow API key.
- `--dataset`: Path to your training dataset directory containing images and annotations.
- `--overwrite`: Overwrite existing dataset (include this flag to enable).
- `--model`: Name of the YOLO model for training (default: `"model/yolov8s.pt"`).
- `--output`: Path to the output directory for saving the trained model and artifacts.
- `--output-format`: Output format for exporting the trained model (choose from available formats).
- `--gpu`: Use GPU for training (include this flag if you have a GPU).
- `--tpu`: Use TPU for training (include this flag if you have a TPU).
- `--epochs`: Number of epochs for training (default: 3).
- `--workers`: Number of workers for training (default: 8).
- `--batch`: Batch size for training (default: 8).
- `--config`: Path to a configuration file (if needed).
- `--verbose`: Enable verbose mode (include this flag to enable).

#### Example:
```
python3 train.py --roboflow-api-key xfrYuTKsBzPt4fTpwX0r --roboflow-project parking-space-cgi5j --roboflow-project-version 1 --gpu --output-format tflite
parking-lot-detection-ufzkx
```

#### Optionally you can use [Roboflow](https://app.roboflow.com/) dataset to train your model
```
python train.py --roboflow-project name-of-your-project [ --api-key your-api-key ]
```

4. The trained weights will be saved in the `weights` folder by default.

## Predicting results
To predict data use command line:
```
python predict.py --source path-to-your-source --model path-to-your-model
```

### Arguments

- `--source`: Path to your training data directory containing images and annotations.
- `--model`: Name of the YOLO model for training (default: `"model/yolov8n.pt"`).
- `--gpu`: Use GPU for training (include this flag if you have a GPU).
- `--tpu`: Use TPU for training (include this flag if you have a TPU).

#### Example:
```
python predict.py --source './datasets/data/videoplayback.mp4' --model './yolov5/runs/detect/train29/weights/best.pt' --no-output --preview --gpu
```

## Results
Include some sample output images or videos demonstrating the performance of the detection system on your dataset.

## Contributing
Contributions to the Parking Lot YOLO Detection Project are welcome! If you find any issues or have ideas to improve the project, feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](link_to_license_file). Feel free to use and modify it according to the terms of the license.