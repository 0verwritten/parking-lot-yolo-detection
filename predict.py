# Import required libraries
from ultralytics import YOLO
import tensorflow as tf
import platform
import numpy as np
import cv2
from colorama import Fore, Style
import argparse
from lib.custom_types import ProcessingUnits
import tensorflow as tf

# List of available export formats
AVALIABLE_EXPORT_FORMATS = ["onnx", "torchscript", "pt", "tflite"]

KEY_ESC = 27

# Function to train a YOLO model
def train_model(
    source: str,
    model_name: str,
    processing_unit: ProcessingUnits,
):
    
    model = YOLO(model_name)
    # # Set processing unit (CPU/GPU/TPU)
    match processing_unit:
        case ProcessingUnits.GPU:
            if not tf.test.gpu_device_name():
                print("!! Warning: GPU not found !!")
            else:
                try:
                    pass
                    # model.to("cuda")
                except Exception:
                    print("!! Warning: GPU delegate not found !!")
        case ProcessingUnits.TPU:
            if platform.machine() == "aarch64" and "USB Accelerator" in platform.uname().release:
                tf.lite.experimental.load_delegate('libedgetpu.so.1')
                pass
                # model.to("edge")
            else:
                print("!! Warning: TPU not found !!")
        case _:
            # model.to("cpu")
            print()

    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Make detections 
        results = model.predict(frame)
        
        cv2.imshow('YOLO', np.squeeze(results[0].plot()))
        pressed_key = cv2.waitKey(10) & 0xFF
        if pressed_key == ord('q') or pressed_key == KEY_ESC:
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
    parser.add_argument("--model", type=str, default="model/yolov8n.pt", help="Name of your yolo model.", metavar="default: model/yolov8n.pt")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training.")
    parser.add_argument("--tpu", action="store_true", help="Use TPU for training.")

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
    )
