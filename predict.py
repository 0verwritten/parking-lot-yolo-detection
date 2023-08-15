# Import required libraries
from matplotlib import pyplot as plt
from ultralytics import YOLO
import tensorflow as tf
import platform
import torch
import numpy as np
import cv2
import os
import colorama
from colorama import Fore, Style
from getpass import getpass
import argparse
from lib.custom_types import ProcessingUnits
import tensorflow as tf

# nterpreter = 
#             interpreter.allocate_tensors()

#             # Input prepare
#             input_details = interpreter.get_input_details()
#             output_details = interpreter.get_output_details()
#             input_shape = input_details[0]['shape']
#             input_data = np.random.random(input_shape).astype(np.float32)

# List of available export formats
AVALIABLE_EXPORT_FORMATS = ["onnx", "torchscript", "pt", "tflite"]

label_map = {
    1: 'CarIn',
    2: 'GateClosed',
    3: 'GateOpened',
    4: 'CarOut',
    5: 'CarParked',
    6: 'SpecialCar',
}

# Function to train a YOLO model
def train_model(
    source: str,
    model_name: str,
    processing_unit: ProcessingUnits,
    epochs: int = 3,
    workers: int = 8,
    batch: int = 8,
    output_path: str = None,
    no_output: bool = True,
    preview: bool = True,
    verbose: bool = False,
):

    # Set processing unit (CPU/GPU/TPU)
    match processing_unit:
        case ProcessingUnits.GPU:
            if not tf.test.gpu_device_name():
                print("!! Warning: GPU not found !!")
            else:
                try:
                    gpu_delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1.0')
                    interpreter = tf.lite.Interpreter(model_path=model_name, experimental_delegates=[gpu_delegate])
                except Exception:
                    print("!! Warning: GPU delegate not found !!")
                    interpreter = tf.lite.Interpreter(model_path=model_name)
        case ProcessingUnits.TPU:
            if platform.machine() == "aarch64" and "USB Accelerator" in platform.uname().release:
                edgetpu_delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
                interpreter = tf.lite.Interpreter(model_path=model_name, experimental_delegates=[edgetpu_delegate])
            else:
                print("!! Warning: TPU not found !!")
                interpreter = tf.lite.Interpreter(model_path=model_name)
        case _:
            tf.config.set_visible_devices([], 'GPU')
            interpreter = tf.lite.Interpreter(model_path=model_name)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def preprocess_image(image, target_size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = image / 255.0
        return np.expand_dims(image, axis=0).astype(np.float32)

    def predict_with_tflite(image):
        input_data = preprocess_image(image,  (640, 640))
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data

    if preview:
        cap = cv2.VideoCapture(source)
        print(cap, cap.isOpened())
        while cap.isOpened():
            ret, frame = cap.read()
            
            output = predict_with_tflite(frame)

            # Process the output as needed
            print(output)
            # Process the output to draw bounding boxes
            for detection in output[0]:
                score = detection[2]
                if score > 0.5:  # Adjust this threshold based on your model
                    class_id = int(detection[1])
                    class_name = label_map[class_id]
                    left = int(detection[3] * frame.shape[1])
                    top = int(detection[4] * frame.shape[0])
                    right = int(detection[5] * frame.shape[1])
                    bottom = int(detection[6] * frame.shape[0])

                    # Draw bounding box and label
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_name}: {score:.2f}', (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            # Display the frame with OpenCV
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


# Entry point of the script
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Predict the data on pretrained model."
    )

    # Define command-line arguments
    parser.add_argument("--source", type=str, help="Path to your data.")
    parser.add_argument("--model", type=str, default="model/yolov8s.pt", help="Name of your yolo model.", metavar="default: model/yolov8s.pt")
    parser.add_argument("--output", type=str, help="Path to output directory.")
    parser.add_argument("--no-output", action="store_true", help="Disable output saving.")
    parser.add_argument("--preview", action="store_true", help="Preview output.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training.")
    parser.add_argument("--tpu", action="store_true", help="Use TPU for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for training.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    # Parse command-line arguments
    args = parser.parse_args()
    
    # Validate GPU/TPU selection
    if args.gpu and args.tpu:
        print(
            f"{Fore.RED}Error: You cannot use both GPU and TPU for training.{Style.RESET_ALL}"
        )
        exit()
    else:
        processing_unit = (
            ProcessingUnits.GPU
            if args.gpu
            else ProcessingUnits.TPU
            if args.tpu
            else ProcessingUnits.CPU
        )

    # Start training the model
    train_model(
        args.source,
        args.model,
        processing_unit,
        args.epochs,
        args.workers,
        args.batch,
        args.output,
        args.no_output,
        args.preview,
        args.verbose
    )
