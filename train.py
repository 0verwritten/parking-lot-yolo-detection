# Import required libraries
from matplotlib import pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow
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

# List of available export formats
AVALIABLE_EXPORT_FORMATS = ["onnx", "torchscript", "pt", "tflite"]

# Function to train a YOLO model
def train_model(
    dataset_path: str,
    config_path: str,
    model_name: str,
    processing_unit: ProcessingUnits,
    epochs: int = 3,
    workers: int = 8,
    batch: int = 8,
    output_path: str = None,
    output_format: str = None,
    verbose: bool = False,
):
    # Print training information
    print(f"Training model with dataset: {dataset_path}")
    print(f"Using config file: {config_path}")
    if verbose:
        print("Verbose mode enabled.")

    # Initialize YOLO model
    model = YOLO(model_name)

    # Set processing unit (CPU/GPU/TPU)
    match processing_unit:
        case ProcessingUnits.GPU:
            if device := ("cuda" if torch.cuda.is_available() else "cpu"):
                model.to(device)
        case ProcessingUnits.TPU:
            if platform.machine() == "aarch64" and "USB Accelerator" in platform.uname().release:
                model.to("edge")
                edgetpu_delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
                interpreter = tf.lite.Interpreter(model_path=args.model_path)
                interpreter.add_delegate(edgetpu_delegate)
        case _:
            model.to("cpu")
    
    # Train the model using provided parameters
    model.train(data=config_path, epochs=epochs, workers=workers, batch=batch)

    # Export the trained model
    model.export(format=output_format or "pt")

# Entry point of the script
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Train a model using specified dataset and config."
    )

    # Define command-line arguments
    parser.add_argument("--roboflow-project", type=str, help="Name of your Roboflow project.")
    parser.add_argument("--roboflow-project-version", type=int, default=1, help="Version of your Roboflow project.")
    parser.add_argument("--roboflow-api-key", type=str, help="Your Roboflow API key.")
    parser.add_argument("--dataset", type=str, help="Path to your dataset.")
    parser.add_argument("--override", action="store_true", help="Override existing dataset.")
    parser.add_argument("--model", type=str, default="model/yolov8s.pt", help="Name of your yolo model.", metavar="default: model/yolov8s.pt")
    parser.add_argument("--output", type=str, help="Path to output directory.")
    parser.add_argument("--output-format", type=str, help=f"Output format, e.g. {', '.join(AVALIABLE_EXPORT_FORMATS)}.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training.")
    parser.add_argument("--tpu", action="store_true", help="Use TPU for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for training.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Validate output format
    if args.output_format and args.output_format not in AVALIABLE_EXPORT_FORMATS:
        print(
            f"{Fore.RED}Error: Invalid output format. Please choose from {', '.join(AVALIABLE_EXPORT_FORMATS)}{Style.RESET_ALL}"
        )
        exit()

    # Handle Roboflow integration
    if args.roboflow_api_key and not args.roboflow_project:
        print(
            f"{Fore.YELLOW}Warning: You must specify a Roboflow project name if you want to use the Roboflow API key.{Style.RESET_ALL}"
        )

    if args.roboflow_project:
        if not args.roboflow_api_key:
            args.roboflow_api_key = getpass(prompt="Enter your Roboflow API key: ")

        roboflow = Roboflow(api_key=args.roboflow_api_key)
        project = roboflow.workspace("worker-lod8r").project("parking-space-cgi5j")
        dataset = project.version(args.roboflow_project_version).download(
            model_format="yolov8",
            location=args.dataset or "./dataset",
            overwrite=args.override,
        )

        args.dataset = dataset.location
        args.config = dataset.location + "/data.yaml"
        if args.verbose:
            print(f"Dataset downloaded to {args.dataset}.")
            print(f"Config downloaded to {args.config}.")

    # Check for required dataset and config
    if not args.dataset or not args.config:
        print(
            f"{Fore.RED}Error: You must specify a dataset and config file or Roboflow project.{Style.RESET_ALL}"
        )
        exit()

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
        args.dataset,
        args.config,
        args.model,
        processing_unit,
        args.epochs,
        args.workers,
        args.batch,
        args.output,
        args.output_format,
        args.verbose
    )
