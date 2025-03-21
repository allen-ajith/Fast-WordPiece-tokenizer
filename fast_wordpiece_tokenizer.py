import re
import os
import json
from typing import Dict, List, Optional, Set, Tuple, Union
import unicodedata
import regex
import numpy as np
from collections import defaultdict
from functools import lru_cache
import multiprocessing
from tqdm import tqdm


class FastWordPieceTokenizer:
    """
    High-performance WordPiece tokenizer optimized for BERT tokenization.
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        max_input_chars_per_word: int = 100,
        remove_emojis: bool = False,
        remove_accents: bool = True,
        lowercase: bool = True,
        strip_chars: Optional[str] = None,
        do_basic_tokenize: bool = True,
        never_split: Optional[List[str]] = None,
        cache_size: int = 100000
    ):
        self.max_input_chars_per_word = max_input_chars_per_word
        self.remove_emojis = remove_emojis
        self.remove_accents = remove_accents
        self.lowercase = lowercase
        self.strip_chars = strip_chars
        self.do_basic_tokenize = do_basic_tokenize
        self.never_split = set(never_split if never_split is not None else [])
        self.cache_size = cache_size
        
        # Special tokens
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.special_tokens = {self.unk_token, self.sep_token, self.pad_token, 
                              self.cls_token, self.mask_token}
        
        # Set up vocabulary
        if vocab is not None:
            self.vocab = vocab
        elif vocab_file is not None:
            self.vocab = self._load_vocab(vocab_file)
        else:
            raise ValueError("Either vocab_file or vocab must be provided")
        
        # Pre-compute ID for unknown token
        self.unk_token_id = self.vocab.get(self.unk_token)
        
        # Use numpy array for faster id lookups
        max_id = max(self.vocab.values())
        if max_id < 1_000_000:
            self.id_array = np.zeros(max_id + 1, dtype=np.object_)
            for k, v in self.vocab.items():
                self.id_array[v] = k
        else:
            self.id_array = None
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Precompile regex patterns
        self._setup_regex_patterns()
        
        # Create trie dictionary
        self._create_trie()
        
        # Set up caching
        self._setup_caching()
        
        # CPU cores for parallelization
        self.num_cores = min(8, multiprocessing.cpu_count() - 2)
    
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        vocab = {}
        total_lines = sum(1 for _ in open(vocab_file, 'r', encoding='utf-8'))
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=total_lines, desc="Loading vocabulary", disable=total_lines < 10000)):
                token = line.rstrip('\n')
                vocab[token] = i
        return vocab
    
    def _setup_regex_patterns(self) -> None:
        self.basic_tokenizer_pattern = re.compile(r'(\s+|[^\w\s]+)')
        
        if self.remove_emojis:
            self.emoji_pattern = regex.compile(
                r'['
                r'\U0001F600-\U0001F64F'  # emoticons
                r'\U0001F300-\U0001F5FF'  # symbols & pictographs
                r'\U0001F680-\U0001F6FF'  # transport & map symbols
                r'\U0001F700-\U0001F77F'  # alchemical symbols
                r'\U0001F780-\U0001F7FF'  # Geometric Shapes
                r'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
                r'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
                r'\U0001FA00-\U0001FA6F'  # Chess Symbols
                r'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
                r'\U00002702-\U000027B0'  # Dingbats
                r'\U000024C2-\U0000257F'  # Enclosed characters
                r']+'
            )
    
    def _create_trie(self) -> None:
        show_progress = len(self.vocab) > 10000
        
        # Main trie for full words
        self.trie = {}
        full_words = [word for word in self.vocab if not word.startswith('##') and word not in self.special_tokens]
        
        word_iter = tqdm(full_words, desc="Building word trie") if show_progress else full_words
        for word in word_iter:
            current = self.trie
            for char in word:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['*'] = True
        
        self.first_char_set = set(self.trie.keys())
        
        # Subword trie
        self.subword_trie = {}
        subwords = [word[2:] for word in self.vocab if word.startswith('##')]
        
        subword_iter = tqdm(subwords, desc="Building subword trie") if show_progress else subwords
        for word in subword_iter:
            current = self.subword_trie
            for char in word:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['*'] = True
            
        self.first_subword_char_set = set(self.subword_trie.keys())
    
    def _setup_caching(self) -> None:
        self._preprocess_text = lru_cache(maxsize=self.cache_size)(self._preprocess_text_impl)
        self._fast_wordpiece_tokenize = lru_cache(maxsize=self.cache_size)(self._fast_wordpiece_tokenize_impl)
        self._basic_tokenize = lru_cache(maxsize=self.cache_size)(self._basic_tokenize_impl)
    
    def save_vocab(self, vocab_path: str) -> None:
        with open(vocab_path, 'w', encoding='utf-8') as f:
            items = sorted(self.vocab.items(), key=lambda x: x[1])
            items_iter = tqdm(items, desc="Saving vocabulary") if len(items) > 10000 else items
            for token, token_id in items_iter:
                f.write(token + '\n')
    
    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        
        vocab_path = os.path.join(save_directory, "vocab.txt")
        self.save_vocab(vocab_path)
        
        config = {
            "unk_token": self.unk_token,
            "sep_token": self.sep_token,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "mask_token": self.mask_token,
            "max_input_chars_per_word": self.max_input_chars_per_word,
            "remove_emojis": self.remove_emojis,
            "remove_accents": self.remove_accents,
            "lowercase": self.lowercase,
            "strip_chars": self.strip_chars,
            "do_basic_tokenize": self.do_basic_tokenize,
            "never_split": list(self.never_split),
            "cache_size": self.cache_size
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, directory: str):
        vocab_path = os.path.join(directory, "vocab.txt")
        config_path = os.path.join(directory, "tokenizer_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        return cls(vocab_file=vocab_path, **config)
    
    def _preprocess_text_impl(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_accents:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        if self.remove_emojis:
            text = self.emoji_pattern.sub('', text)
        
        if self.strip_chars:
            trans_table = str.maketrans('', '', self.strip_chars)
            text = text.translate(trans_table)
        
        return text
    
    def _basic_tokenize_impl(self, text: str) -> List[str]:
        if self.never_split:
            replacement_needed = any(token in text for token in self.never_split)
                    
            if replacement_needed:
                pattern = '|'.join(re.escape(token) for token in self.never_split if token in text)
                if pattern:
                    pattern = re.compile(f'({pattern})')
                    parts = []
                    last_end = 0
                    for match in pattern.finditer(text):
                        start, end = match.span()
                        if start > last_end:
                            parts.append(text[last_end:start])
                        parts.append(f" {match.group()} ")
                        last_end = end
                    if last_end < len(text):
                        parts.append(text[last_end:])
                    text = ''.join(parts)
        
        tokens = [token for token in self.basic_tokenizer_pattern.split(text) 
                 if token and not token.isspace()]
        
        return tokens
    
    def _fast_wordpiece_tokenize_impl(self, word: str) -> List[str]:
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]
        
        if word in self.vocab:
            return [word]
        
        if word in self.never_split:
            return [word]
        
        if word and word[0] not in self.first_char_set:
            return [self.unk_token]
            
        tokens = []
        start = 0
        length = len(word)
        
        while start < length:
            if start == 0:
                found_match = False
                current = self.trie
                longest_end = start
                
                if word[start] not in current:
                    return [self.unk_token]
                
                for i in range(start, length):
                    char = word[i]
                    if char in current:
                        current = current[char]
                        if '*' in current:
                            longest_end = i + 1
                            found_match = True
                    else:
                        break
                
                if found_match:
                    tokens.append(word[start:longest_end])
                    start = longest_end
                    continue
                else:
                    if word[start] in self.vocab:
                        tokens.append(word[start])
                        start += 1
                        continue
                    
                    return [self.unk_token]
            else:
                if start < length and word[start] not in self.first_subword_char_set:
                    return [self.unk_token]
                    
                found_match = False
                current = self.subword_trie
                longest_end = start
                
                for i in range(start, length):
                    char = word[i]
                    if char in current:
                        current = current[char]
                        if '*' in current:
                            longest_end = i + 1
                            found_match = True
                    else:
                        break
                
                if found_match:
                    tokens.append(f"##{word[start:longest_end]}")
                    start = longest_end
                    continue
                else:
                    return [self.unk_token]
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        text = self._preprocess_text(text)
        
        if not text:
            return []
            
        est_tokens = min(1000, len(text) // 2)
        all_tokens = []
        all_tokens_append = all_tokens.append
        all_tokens_extend = all_tokens.extend
        
        if self.do_basic_tokenize:
            words = self._basic_tokenize(text)
        else:
            words = [text]
        
        for word in words:
            if not word:
                continue
                
            if word in self.never_split:
                all_tokens_append(word)
                continue
                
            subwords = self._fast_wordpiece_tokenize(word)
            all_tokens_extend(subwords)
        
        return all_tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        vocab_dict = defaultdict(lambda: self.unk_token_id)
        vocab_dict.update(self.vocab)
        return [vocab_dict[token] for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        if self.id_array is not None:
            valid_indices = np.array(ids) < len(self.id_array)
            result = [self.unk_token] * len(ids)
            
            valid_ids = np.array(ids)[valid_indices]
            if len(valid_ids) > 0:
                valid_tokens = self.id_array[valid_ids]
                
                for i, (is_valid, token) in enumerate(zip(valid_indices, valid_tokens)):
                    if is_valid:
                        result[i] = token
            return result
        else:
            return [self.ids_to_tokens.get(id_, self.unk_token) for id_ in ids]
    
    @lru_cache(maxsize=100000)
    def _cached_encode(self, text: str, add_special_tokens: bool) -> Tuple[int, ...]:
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
            
        return tuple(self.convert_tokens_to_ids(tokens))
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return list(self._cached_encode(text, add_special_tokens))
    
    def _process_batch(self, batch_texts, add_special_tokens):
        return [self.encode(text, add_special_tokens) for text in batch_texts]
    
    def batch_encode(self, texts: List[str], add_special_tokens: bool = True,
                     batch_size: int = 1000, use_parallel: bool = True) -> List[List[int]]:
        total_texts = len(texts)
        results = []
        
        update_interval = max(1, total_texts // 100)
        
        pbar = tqdm(total=total_texts, desc="Encoding texts")
        processed = 0
        
        if use_parallel and self.num_cores > 1 and len(texts) > 1000:
            with multiprocessing.Pool(processes=self.num_cores) as pool:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    sub_batch_size = max(1, len(batch) // self.num_cores)
                    sub_batches = [batch[j:j + sub_batch_size] 
                                  for j in range(0, len(batch), sub_batch_size)]
                    
                    parallel_args = [(sub_batch, add_special_tokens) for sub_batch in sub_batches]
                    batch_results = pool.starmap(self._process_batch, parallel_args)
                    
                    for sub_result in batch_results:
                        results.extend(sub_result)
                    
                    processed += len(batch)
                    if processed // update_interval > (processed - len(batch)) // update_interval or processed == total_texts:
                        pbar.update(processed - pbar.n)
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = [self.encode(text, add_special_tokens) for text in batch]
                results.extend(batch_results)
                
                processed += len(batch)
                if processed // update_interval > (processed - len(batch)) // update_interval or processed == total_texts:
                    pbar.update(processed - pbar.n)
        
        pbar.close()
            
        return results
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        result_parts = []
        space_needed = False
        
        for token in tokens:
            if token.startswith("##"):
                result_parts.append(token[2:])
                space_needed = True
            else:
                if result_parts and space_needed:
                    result_parts.append(" ")
                result_parts.append(token)
                space_needed = True
                
        return "".join(result_parts).strip()

    def batch_decode(self, all_token_ids: List[List[int]], skip_special_tokens: bool = True,
                    batch_size: int = 1000, use_parallel: bool = True) -> List[str]:
        """Decode multiple sequences of token IDs in batches with parallelization."""
        total_sequences = len(all_token_ids)
        results = []
        
        update_interval = max(1, total_sequences // 100)
        pbar = tqdm(total=total_sequences, desc="Decoding sequences")
        processed = 0
        
        def _process_decode_batch(batch_ids):
            return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
        
        if use_parallel and self.num_cores > 1 and total_sequences > 1000:
            with multiprocessing.Pool(processes=self.num_cores) as pool:
                for i in range(0, total_sequences, batch_size):
                    batch = all_token_ids[i:i + batch_size]
                    
                    sub_batch_size = max(1, len(batch) // self.num_cores)
                    sub_batches = [batch[j:j + sub_batch_size] 
                                  for j in range(0, len(batch), sub_batch_size)]
                    
                    batch_results = pool.map(_process_decode_batch, sub_batches)
                    
                    for sub_result in batch_results:
                        results.extend(sub_result)
                    
                    processed += len(batch)
                    if processed // update_interval > (processed - len(batch)) // update_interval:
                        pbar.update(processed - pbar.n)
        else:
            for i in range(0, total_sequences, batch_size):
                batch = all_token_ids[i:i + batch_size]
                batch_results = _process_decode_batch(batch)
                results.extend(batch_results)
                
                processed += len(batch)
                if processed // update_interval > (processed - len(batch)) // update_interval:
                    pbar.update(processed - pbar.n)
        
        pbar.close()
        return results


def concatenate_files_to_single(file_paths, output_path=None, encoding='utf-8'):
    """Concatenate multiple files into a single file."""
    total_size = sum(os.path.getsize(path) for path in file_paths)
    
    with open(output_path or 'combined_corpus.txt', 'w', encoding=encoding) as outfile:
        with tqdm(total=total_size, desc="Concatenating files") as pbar:
            for file_path in file_paths:
                with open(file_path, 'r', encoding=encoding) as infile:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = infile.read(chunk_size)
                        if not chunk:
                            break
                        outfile.write(chunk)
                        pbar.update(len(chunk.encode(encoding)))
                outfile.write('\n')  # Separate files with newline
    
    if output_path:
        return output_path
    else:
        return 'combined_corpus.txt'


def train_wordpiece_vocab(
    files: List[str],
    vocab_size: int = 30000,
    min_frequency: int = 2,
    special_tokens: List[str] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    lowercase: bool = True,
    remove_accents: bool = True
) -> Dict[str, int]:
    """
    Train a WordPiece vocabulary from files.
    Concatenates files into a single corpus before processing.
    """
    combined_file = concatenate_files_to_single(files, 'combined_corpus.txt')
    
    # Process the combined file
    print(f"Processing combined corpus file...")
    
    # Pre-compile patterns
    word_pattern = re.compile(r'\b\w+\b|[^\w\s]')
    
    # Count words and characters
    word_counts = defaultdict(int)
    char_vocab = set()
    
    # Process in chunks
    chunk_size = 10 * 1024 * 1024  # 10MB
    file_size = os.path.getsize(combined_file)
    
    with open(combined_file, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, desc="Processing corpus") as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Preprocess chunk
                if lowercase:
                    chunk = chunk.lower()
                if remove_accents:
                    chunk = unicodedata.normalize('NFKD', chunk)
                    chunk = ''.join([c for c in chunk if not unicodedata.combining(c)])
                
                # Count words and collect characters
                for word in word_pattern.findall(chunk):
                    word_counts[word] += 1
                    char_vocab.update(word)
                
                pbar.update(len(chunk.encode('utf-8')))
    
    # Filter by minimum frequency
    filtered_words = {word: count for word, count in word_counts.items() 
                     if count >= min_frequency}
    
    print(f"Found {len(filtered_words):,} unique words with frequency >= {min_frequency}")
    
    # Create initial vocabulary with special tokens
    vocab = {token: i for i, token in enumerate(special_tokens)}
    token_id = len(vocab)
    
    # Sort words by frequency
    sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
    
    # Add whole words (up to half of remaining slots)
    remaining_slots = vocab_size - len(vocab)
    whole_word_limit = min(remaining_slots // 2, len(sorted_words))
    
    print(f"Adding {whole_word_limit:,} whole words to vocabulary...")
    for i in range(whole_word_limit):
        word, _ = sorted_words[i]
        if word not in vocab:
            vocab[word] = token_id
            token_id += 1
    
    # Add all characters
    for char in sorted(char_vocab):
        if char not in vocab:
            vocab[char] = token_id
            token_id += 1
    
    # Count subword occurrences
    subword_stats = defaultdict(int)
    
    print("Counting subword frequencies...")
    for word, count in tqdm(sorted_words):
        if len(word) <= 1:
            continue
            
        for start in range(len(word)):
            for end in range(start + 1, min(len(word) + 1, start + 20)):
                subword = word[start:end]
                if len(subword) >= 2:
                    if start > 0:
                        subword = f"##{subword}"
                    subword_stats[subword] += count
    
    # Sort subwords by frequency
    sorted_subwords = sorted(subword_stats.items(), key=lambda x: x[1], reverse=True)
    
    # Add top subwords to vocabulary
    remaining_slots = vocab_size - len(vocab)
    subword_limit = min(remaining_slots, len(sorted_subwords))
    
    print(f"Adding {subword_limit:,} subwords to vocabulary...")
    for i in range(subword_limit):
        subword, _ = sorted_subwords[i]
        if subword not in vocab:
            vocab[subword] = token_id
            token_id += 1
            
            if len(vocab) >= vocab_size:
                break
    
    print(f"Final vocabulary size: {len(vocab):,}")
    return vocab


if __name__ == "__main__":
    # Example usage
    # corpus_files = ["file1.txt", "file2.txt", "file3.txt"]
    # vocab = train_wordpiece_vocab(files=corpus_files, vocab_size=30000)
    
    # tokenizer = FastWordPieceTokenizer(vocab=vocab)
    # tokenizer.save_pretrained("bert_tokenizer")
    
    # loaded_tokenizer = FastWordPieceTokenizer.from_pretrained("bert_tokenizer")
    
    # Batch encoding example
    # texts = ["Hello world", "This is another test", "WordPiece is fast"]
    # encoded = loaded_tokenizer.batch_encode(texts)
    # decoded = loaded_tokenizer.batch_decode(encoded)
    
    print("FastWordPieceTokenizer implementation complete.")