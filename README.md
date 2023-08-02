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
To train the YOLO model on your custom dataset, follow these steps:

1. Organize your dataset as described in the [Dataset](#dataset) section.

2. Modify the `./dataset/Parking.../config.yaml` file to set the training parameters, such as batch size, learning rate, etc.

3. Start the training process using the following command:

```
python train.py --dataset path/to/your/dataset/ --config config/config.yaml
```

4. The trained weights will be saved in the `weights` folder by default.

## Testing
To evaluate the performance of the trained YOLO model, run the following command:

```
python test.py --dataset path/to/your/dataset/ --weights path/to/your/trained/weights
```

## Results
Include some sample output images or videos demonstrating the performance of the detection system on your dataset.

## Contributing
Contributions to the Parking Lot YOLO Detection Project are welcome! If you find any issues or have ideas to improve the project, feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](link_to_license_file). Feel free to use and modify it according to the terms of the license.