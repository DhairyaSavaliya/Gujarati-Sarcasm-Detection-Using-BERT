import sys
import argparse
from src.preprocess import preprocess_dataset
from src.train import train_model
from src.predict import predict

def main():
    parser = argparse.ArgumentParser(description="Gujarati Sarcasm Detection using MuRIL BERT")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preprocess command
    subparsers.add_parser("preprocess", help="Preprocess the raw dataset")

    # Train command
    subparsers.add_parser("train", help="Train the MuRIL model on preprocessed data")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict sarcasm for a given Gujarati text")
    predict_parser.add_argument("text", type=str, help="Gujarati text to check for sarcasm")

    args = parser.parse_args()

    if args.command == "preprocess":
        print("Starting preprocessing...")
        preprocess_dataset()
        print("Preprocessing completed!")
    elif args.command == "train":
        print("Starting training...")
        train_model()
        print("Training completed!")
    elif args.command == "predict":
        result = predict(args.text)
        print(f"\nInput Text: {args.text}")
        print(f"Prediction: {result}\n")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
