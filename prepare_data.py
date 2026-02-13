"""
Prepare text for fine-tuning with MLX.

This script converts your novel into the JSONL format needed for training.
It creates training examples by chunking your text into overlapping segments.

Usage:
    python prepare_data.py --file novel.txt --output training_data
"""

import json
import argparse
import os
from pathlib import Path


def chunk_text(text, chunk_size=512, overlap=64):
    """
    Split text into overlapping chunks.
    
    Args:
        text: The full text to chunk
        chunk_size: Target size of each chunk (in characters)
        overlap: How much chunks should overlap
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to end at a sentence boundary
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for punct in ['. ', '.\n', '? ', '!\n', '?\n', '!\n']:
                last_punct = text[start:end].rfind(punct)
                if last_punct > chunk_size * 0.5:  # At least half the chunk
                    end = start + last_punct + len(punct)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def create_training_examples(chunks, style="continuation"):
    """
    Create training examples from chunks.
    
    Args:
        chunks: List of text chunks
        style: Type of training examples to create
            - "continuation": Simple text continuation
            - "writing": "Write in my style about X" format
    
    Returns:
        List of training examples (dicts)
    """
    examples = []
    
    if style == "continuation":
        # Simple: just the text itself
        for chunk in chunks:
            examples.append({"text": chunk})
    
    elif style == "writing":
        # Instruction format: teach it to write like you
        for chunk in chunks:
            # Take first sentence as "topic"
            first_sentence_end = chunk.find('. ')
            if first_sentence_end > 10:
                topic = chunk[:first_sentence_end + 1]
                continuation = chunk[first_sentence_end + 2:]
                
                examples.append({
                    "messages": [
                        {"role": "user", "content": f"Continue this text in the same style:\n\n{topic}"},
                        {"role": "assistant", "content": continuation}
                    ]
                })
            else:
                examples.append({"text": chunk})
    
    return examples


def save_jsonl(examples, filepath):
    """Save examples as JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Prepare text for fine-tuning')
    parser.add_argument('--file', type=str, required=True, help='Path to your text file')
    parser.add_argument('--output', type=str, default='training_data', help='Output directory')
    parser.add_argument('--chunk_size', type=int, default=512, help='Size of text chunks')
    parser.add_argument('--style', type=str, default='continuation', 
                        choices=['continuation', 'writing'],
                        help='Training style')
    parser.add_argument('--val_split', type=float, default=0.1, 
                        help='Fraction of data for validation')
    args = parser.parse_args()
    
    # Read text
    print(f"Reading {args.file}...")
    with open(args.file, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters")
    
    # Chunk text
    print(f"Chunking with size={args.chunk_size}...")
    chunks = chunk_text(text, chunk_size=args.chunk_size)
    print(f"Created {len(chunks)} chunks")
    
    # Create examples
    examples = create_training_examples(chunks, style=args.style)
    print(f"Created {len(examples)} training examples")
    
    # Split into train/validation
    split_idx = int(len(examples) * (1 - args.val_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Save files
    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'valid.jsonl'
    
    save_jsonl(train_examples, train_path)
    save_jsonl(val_examples, val_path)
    
    print(f"\nSaved:")
    print(f"  Training:   {train_path} ({len(train_examples)} examples)")
    print(f"  Validation: {val_path} ({len(val_examples)} examples)")
    
    # Show sample
    print(f"\nSample training example:")
    print("-" * 40)
    sample = train_examples[0]
    if 'text' in sample:
        print(sample['text'][:300] + "...")
    else:
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")
    print("-" * 40)


if __name__ == "__main__":
    main()
