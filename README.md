
# Fast WordPiece Tokenizer

## Overview
This repository contains my implementation of **Fast WordPiece Tokenization**, based on the paper "[Fast WordPiece Tokenization](https://ar5iv.labs.arxiv.org/html/2012.15524)". It is designed for efficient tokenization using trie-based matching, caching, and parallel batch processing.

---

## Files

### `fast_wordpiece_tokenizer.py`
- **Description**: Implements the `FastWordPieceTokenizer` class.
  
  - Trie-based token lookup for linear-time tokenization.
  - LRU caching for repeated operations.
  - Batch encoding and decoding with multiprocessing support.
  - Customizable preprocessing options (e.g., removing emojis, accents, or lowercasing text).
- **Usage**:
  - Tokenize text into subwords.
  - Encode text into token IDs and decode token IDs back into text.
  - Save and load pretrained tokenizer configurations.

---

### `tokenizer_trainer.py`
- **Description**: Script for training a WordPiece vocabulary from text corpora and creating a tokenizer.
  
  - Supports customizable parameters such as vocabulary size, minimum frequency, and special tokens.
  - Includes functionality to test trained tokenizers with sample inputs.
- **Usage**:
  - Train a new WordPiece tokenizer from a directory of text files.
  - Save the trained tokenizer and vocabulary for later use.

---

### `jsontotxt.py`
- **Description**: Utility script to convert JSON files into plain text format for tokenizer training.
  
  - Reads JSON lines from input files and writes them as formatted plain text.
- **Usage**:
  - Convert JSON datasets into `.txt` files to prepare training data for the tokenizer.


