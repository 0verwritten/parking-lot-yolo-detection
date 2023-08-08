# Import required libraries
from matplotlib import pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow
import numpy as np
import cv2
import os
import colorama
from colorama import Fore, Style
from getpass import getpass
import argparse
# from lib.custom_types import ProcessingUnits

# nterpreter = tflite.Interpreter(model_path='model.tflite',
#                                              experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
#             interpreter.allocate_tensors()

#             # Input prepare
#             input_details = interpreter.get_input_details()
#             output_details = interpreter.get_output_details()
#             input_shape = input_details[0]['shape']
#             input_data = np.random.random(input_shape).astype(np.float32)

# List of available export formats
AVALIABLE_EXPORT_FORMATS = ["onnx", "torchscript", "pt", "tflite"]

# Function to train a YOLO model
def train_model(
    source: str,
    model_name: str,
    # processing_unit: ProcessingUnits,
    epochs: int = 3,
    workers: int = 8,
    batch: int = 8,
    output_path: str = None,
    no_output: bool = True,
    preview: bool = True,
    verbose: bool = False,
):
    # Initialize YOLO model
    model = YOLO(model_name)

    # Set processing unit (CPU/GPU/TPU)
    # model.to("cuda:0")
    # match processing_unit:
    #     case ProcessingUnits.GPU:
    #     case ProcessingUnits.TPU:
    #         pass
    #     case _:
    #         model.to("cpu:0")

    if preview:
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Make detections 
            # results = model(frame)
            results = model.predict(frame)
            print(results)
            
            cv2.imshow('YOLO', np.squeeze(results[0].plot()))
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    if not no_output:
        model.predict(source, output=output_path or "output", save_txt=True, save_conf=True, save_crop=True, save_bbox=True, save_img=True, augment=True, verbose=verbose)

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
        pass
        # processing_unit = (
        #     ProcessingUnits.GPU
        #     if args.gpu
        #     else ProcessingUnits.TPU
        #     if args.tpu
        #     else ProcessingUnits.CPU
        # )

    # Start training the model
    train_model(
        args.source,
        args.model,
        # processing_unit,
        args.epochs,
        args.workers,
        args.batch,
        args.output,
        args.no_output,
        args.preview,
        args.verbose
    )
