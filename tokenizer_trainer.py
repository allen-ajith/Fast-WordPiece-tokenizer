#!/usr/bin/env python
import os
import argparse
import glob
from typing import List
import sys

# Import the FastWordPieceTokenizer and related functions
from fast_wordpiece_tokenizer import FastWordPieceTokenizer, train_wordpiece_vocab

def get_input_with_default(prompt: str, default: str) -> str:
    """Get input from user with a default value."""
    user_input = input(f"{prompt} [{default}]: ").strip()
    return user_input if user_input else default

def confirm_action(message: str) -> bool:
    """Confirm an action with the user."""
    response = input(f"{message} (y/n): ").strip().lower()
    return response in ('y', 'yes')

def get_text_files(directory: str) -> List[str]:
    """Get all text files in a directory and its subdirectories."""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
        
    text_files = []
    
    # Look for .txt files
    txt_files = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
    if txt_files:
        text_files.extend(txt_files)
    
    # Also look for .text files
    text_ext_files = glob.glob(os.path.join(directory, "**/*.text"), recursive=True)
    if text_ext_files:
        text_files.extend(text_ext_files)
        
    if not text_files:
        print(f"Warning: No .txt or .text files found in '{directory}'.")
        
        # Ask if user wants to specify pattern
        if confirm_action("Do you want to specify a file pattern?"):
            pattern = input("Enter file pattern (e.g., '**/*.json'): ").strip()
            custom_files = glob.glob(os.path.join(directory, pattern), recursive=True)
            if custom_files:
                text_files.extend(custom_files)
                print(f"Found {len(custom_files)} files with pattern '{pattern}'.")
            else:
                print(f"No files found with pattern '{pattern}'.")
                sys.exit(1)
        else:
            sys.exit(1)
    
    return text_files

def test_tokenizer(tokenizer):
    """Test the tokenizer with sample text input from user."""
    while True:
        test_text = input("\nEnter a sample text (or 'exit' to quit testing): ").strip()
        if test_text.lower() == 'exit':
            break
            
        tokens = tokenizer.tokenize(test_text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        print("\nTokenization result:")
        print(f"Text: '{test_text}'")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: '{tokenizer.decode(token_ids)}'")

def main():
    print("=" * 60)
    print("   WordPiece Tokenizer Training Script")
    print("=" * 60)
    
    # Ask if user wants to test an existing tokenizer
    if confirm_action("Do you want to test an existing trained tokenizer?"):
        tokenizer_dir = input("Enter path to existing tokenizer directory: ").strip()
        if not os.path.exists(tokenizer_dir):
            print(f"Error: Directory '{tokenizer_dir}' does not exist.")
            return
            
        try:
            tokenizer = FastWordPieceTokenizer.from_pretrained(tokenizer_dir)
            print(f"Loaded tokenizer from '{tokenizer_dir}'")
            test_tokenizer(tokenizer)
            
            if confirm_action("Do you want to train a new tokenizer?"):
                # Continue with training
                pass
            else:
                return
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            if not confirm_action("Do you want to train a new tokenizer instead?"):
                return
    
    # Get data directory
    data_dir = input("Enter path to data directory: ").strip()
    
    # Get text files
    text_files = get_text_files(data_dir)
    print(f"Found {len(text_files)} text files.")
    
    # List first few files
    if len(text_files) > 0:
        sample_size = min(5, len(text_files))
        print(f"First {sample_size} files:")
        for i in range(sample_size):
            print(f"  - {text_files[i]}")
        
        if len(text_files) > sample_size:
            print(f"  ... and {len(text_files) - sample_size} more")
    
    # Get parameters with BERT defaults
    print("\nEnter training parameters (press Enter to use defaults):")
    vocab_size = int(get_input_with_default("Vocabulary size", "30522"))
    min_frequency = int(get_input_with_default("Minimum token frequency", "2"))
    lowercase = get_input_with_default("Lowercase text (true/false)", "true").lower() == "true"
    remove_accents = get_input_with_default("Remove accents (true/false)", "true").lower() == "true"
    remove_emojis = get_input_with_default("Remove emojis (true/false)", "true").lower() == "true"
    
    # Special tokens
    default_special_tokens = "[PAD],[UNK],[CLS],[SEP],[MASK]"
    special_tokens_input = get_input_with_default("Special tokens (comma-separated)", default_special_tokens)
    special_tokens = [token.strip() for token in special_tokens_input.split(",")]
    
    # Output directory
    output_dir = get_input_with_default("Output directory", "wordpiece_tokenizer")
    
    # Confirm settings
    print("\nTraining Configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Number of text files: {len(text_files)}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Minimum token frequency: {min_frequency}")
    print(f"  Lowercase: {lowercase}")
    print(f"  Remove accents: {remove_accents}")
    print(f"  Remove emojis: {remove_emojis}")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Output directory: {output_dir}")
    
    if not confirm_action("\nProceed with training?"):
        print("Training cancelled.")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Train vocabulary
    print("\nTraining vocabulary... (this may take a while)")
    vocab = train_wordpiece_vocab(
        files=text_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        lowercase=lowercase,
        remove_accents=remove_accents
    )
    
    # Initialize and save tokenizer
    print(f"Creating tokenizer with {len(vocab)} tokens...")
    tokenizer = FastWordPieceTokenizer(
        vocab=vocab,
        unk_token=special_tokens[1] if len(special_tokens) > 1 else "[UNK]",
        sep_token=special_tokens[3] if len(special_tokens) > 3 else "[SEP]",
        pad_token=special_tokens[0] if len(special_tokens) > 0 else "[PAD]",
        cls_token=special_tokens[2] if len(special_tokens) > 2 else "[CLS]",
        mask_token=special_tokens[4] if len(special_tokens) > 4 else "[MASK]",
        lowercase=lowercase,
        remove_accents=remove_accents,
        remove_emojis=remove_emojis
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to '{output_dir}'")
    
    # Test the tokenizer
    if confirm_action("Do you want to test the tokenizer with sample texts?"):
        test_tokenizer(tokenizer)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")