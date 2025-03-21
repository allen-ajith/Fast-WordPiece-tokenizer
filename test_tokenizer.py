import os
import time
import json
from fast_wordpiece_tokenizer import FastWordPieceTokenizer

def test_tokenizer():
    print("\n" + "="*50)
    print("FAST WORDPIECE TOKENIZER TEST SCRIPT")
    print("="*50 + "\n")
    
    # Define sample vocabulary for testing
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1, 
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "the": 5,
        "quick": 6,
        "brown": 7,
        "fox": 8,
        "jumps": 9,
        "over": 10,
        "lazy": 11,
        "dog": 12,
        "hello": 13,
        "world": 14,
        "a": 15,
        "b": 16,
        "c": 17,
        "##ing": 18,
        "##ed": 19,
        "##s": 20,
        "test": 21,
        "this": 22,
        "is": 23,
        "##ing": 24,
        "walk": 25,
        "talk": 26,
        "run": 27,
        "##ick": 28,
        "qu": 29
    }
    
    # Create the tokenizer
    print("Initializing tokenizer...")
    tokenizer = FastWordPieceTokenizer(
        vocab=vocab,
        max_input_chars_per_word=100,
        remove_accents=True,
        lowercase=True
    )
    
    # Test basic tokenization
    print("\n1. Testing basic tokenization...")
    sample_text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.tokenize(sample_text)
    print(f"Text: '{sample_text}'")
    print(f"Tokens: {tokens}")
    
    # Test token to ids conversion
    print("\n2. Testing token to ids conversion...")
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")
    
    # Test ids to tokens conversion
    print("\n3. Testing ids to tokens conversion...")
    tokens_back = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"Tokens (converted back): {tokens_back}")
    
    # Test encode method
    print("\n4. Testing encode method...")
    encoded = tokenizer.encode(sample_text, add_special_tokens=True)
    print(f"Encoded (with special tokens): {encoded}")
    encoded = tokenizer.encode(sample_text, add_special_tokens=False)
    print(f"Encoded (without special tokens): {encoded}")
    
    # Test decode method
    print("\n5. Testing decode method...")
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"Decoded: '{decoded}'")
    
    # Test batch encode
    print("\n6. Testing batch encode...")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test.",
        "Testing batch processing with multiple sentences."
    ]
    start_time = time.time()
    batch_encoded = tokenizer.batch_encode(texts, add_special_tokens=True, use_parallel=False)
    single_time = time.time() - start_time
    print(f"Batch encoded (sequential): {batch_encoded}")
    print(f"Time taken (sequential): {single_time:.4f} seconds")
    
    # Test batch encode with parallelization
    start_time = time.time()
    batch_encoded_parallel = tokenizer.batch_encode(texts, add_special_tokens=True, use_parallel=True)
    parallel_time = time.time() - start_time
    print(f"Time taken (parallel): {parallel_time:.4f} seconds")
    
    # Test batch decode
    print("\n7. Testing batch decode...")
    batch_decoded = tokenizer.batch_decode(batch_encoded, skip_special_tokens=True)
    print(f"Batch decoded: {batch_decoded}")
    
    # Test saving and loading
    print("\n8. Testing save and load functions...")
    save_dir = "test_tokenizer"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")
    
    loaded_tokenizer = FastWordPieceTokenizer.from_pretrained(save_dir)
    print("Tokenizer loaded successfully")
    
    # Verify loaded tokenizer works
    loaded_tokens = loaded_tokenizer.tokenize(sample_text)
    print(f"Tokens from loaded tokenizer: {loaded_tokens}")
    
    # Test batch_tokenize_dict
    print("\n9. Testing batch_tokenize_dict...")
    
    # First, add the method to the tokenizer class
    FastWordPieceTokenizer.batch_tokenize_dict = lambda self, *args, **kwargs: __import__('fast_wordpiece_tokenizer').batch_tokenize_dict(self, *args, **kwargs)
    
    # Test with single text
    single_tokenized = tokenizer.batch_tokenize_dict(
        sample_text, 
        add_special_tokens=True,
        return_offsets=True
    )
    print("\nSingle text tokenization:")
    print(json.dumps(single_tokenized, indent=2))
    
    # Test with multiple texts
    multi_tokenized = tokenizer.batch_tokenize_dict(
        texts,
        add_special_tokens=True,
        return_offsets=True,
        padding=True,
        max_length=20,
        use_parallel=False
    )
    print("\nMultiple texts tokenization (first entry):")
    print(json.dumps(multi_tokenized[0], indent=2))
    
    # Test with padding and truncation
    print("\n10. Testing padding and truncation with batch_tokenize_dict...")
    long_text = "The quick brown fox jumps over the lazy dog. " * 10
    padded_tokenized = tokenizer.batch_tokenize_dict(
        long_text,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=15
    )
    print(f"Original text length: {len(long_text)}")
    print(f"Truncated token IDs length: {len(padded_tokenized['input_ids'])}")
    print(f"Token IDs: {padded_tokenized['input_ids']}")
    print(f"Attention mask: {padded_tokenized['attention_mask']}")
    
    # Test handling of OOV (out of vocabulary) words
    print("\n11. Testing OOV handling...")
    oov_text = "Supercalifragilisticexpialidocious is a very long word."
    oov_tokens = tokenizer.tokenize(oov_text)
    print(f"Text with OOV: '{oov_text}'")
    print(f"Tokens: {oov_tokens}")
    
    # Test unicode and special character handling
    print("\n12. Testing unicode and special character handling...")
    unicode_text = "Café au lait costs €3.50 and has 100% flavor."
    unicode_tokens = tokenizer.tokenize(unicode_text)
    print(f"Unicode text: '{unicode_text}'")
    print(f"Tokens: {unicode_tokens}")
    
    # Performance test on larger text
    print("\n13. Testing performance on larger text...")
    large_text = "The quick brown fox jumps over the lazy dog. " * 1000
    start_time = time.time()
    large_encoded = tokenizer.encode(large_text)
    time_taken = time.time() - start_time
    print(f"Text length: {len(large_text)} characters")
    print(f"Token count: {len(large_encoded)}")
    print(f"Time taken: {time_taken:.4f} seconds")
    print(f"Tokens per second: {len(large_encoded)/time_taken:.2f}")
    
    print("\n" + "="*50)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    test_tokenizer()