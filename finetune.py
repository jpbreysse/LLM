"""
Fine-tune a language model on your text using MLX (Apple Silicon optimized).

This script fine-tunes a pre-trained model on your novel to learn your writing style.

Prerequisites:
    pip install mlx mlx-lm

Usage:
    # Step 1: Download and convert a model (do this once)
    mlx_lm.convert --hf-path mistralai/Mistral-7B-v0.1 -q --mlx-path ./mistral-7b-mlx
    
    # Step 2: Prepare your data
    python prepare_data.py --file novel.txt --output training_data
    
    # Step 3: Fine-tune
    python finetune.py --model ./mistral-7b-mlx --data ./training_data

Alternative smaller models (faster training, less RAM):
    # Phi-3 Mini (3.8B) - Good quality, very fast
    mlx_lm.convert --hf-path microsoft/Phi-3-mini-4k-instruct -q --mlx-path ./phi3-mlx
    
    # TinyLlama (1.1B) - Very fast, lower quality
    mlx_lm.convert --hf-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 -q --mlx-path ./tinyllama-mlx
"""

import subprocess
import argparse
import os
from pathlib import Path


def run_finetuning(model_path, data_path, output_path, config):
    """
    Run MLX fine-tuning using LoRA (Low-Rank Adaptation).
    
    LoRA only trains a small number of additional parameters,
    making it fast and memory-efficient.
    """
    
    cmd = [
        "mlx_lm.lora",
        "--model", model_path,
        "--data", data_path,
        "--train",
        "--adapter-path", output_path,
        "--iters", str(config['iters']),
        "--batch-size", str(config['batch_size']),
        "--learning-rate", str(config['learning_rate']),
        "--lora-layers", str(config['lora_layers']),
    ]
    
    print("=" * 60)
    print("FINE-TUNING WITH MLX")
    print("=" * 60)
    print(f"Model:      {model_path}")
    print(f"Data:       {data_path}")
    print(f"Output:     {output_path}")
    print(f"Iterations: {config['iters']}")
    print(f"Batch size: {config['batch_size']}")
    print("=" * 60)
    print("\nRunning command:")
    print(" ".join(cmd))
    print("\n")
    
    # Run the training
    subprocess.run(cmd, check=True)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nAdapter saved to: {output_path}")
    print(f"\nTo generate text, run:")
    print(f"  python generate_finetuned.py --model {model_path} --adapter {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a model with MLX')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to MLX model directory')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data directory (with train.jsonl)')
    parser.add_argument('--output', type=str, default='./adapters',
                        help='Where to save the fine-tuned adapter')
    parser.add_argument('--iters', type=int, default=1000,
                        help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (reduce if out of memory)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--lora_layers', type=int, default=16,
                        help='Number of layers to apply LoRA to')
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("\nDownload a model first:")
        print("  mlx_lm.convert --hf-path mistralai/Mistral-7B-v0.1 -q --mlx-path ./mistral-7b-mlx")
        return
    
    train_file = Path(args.data) / "train.jsonl"
    if not train_file.exists():
        print(f"Error: Training data not found at {train_file}")
        print("\nPrepare your data first:")
        print("  python prepare_data.py --file novel.txt --output training_data")
        return
    
    config = {
        'iters': args.iters,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lora_layers': args.lora_layers,
    }
    
    run_finetuning(args.model, args.data, args.output, config)


if __name__ == "__main__":
    main()
