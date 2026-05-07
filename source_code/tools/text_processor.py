"""
Text Processor - From Words to Vectors (v2.1)

This version replaces the naive `hash(token)` with a more robust SHA-256 hash
to minimize collisions and ensure a better distribution of token features across
the vector dimensions.
"""
import re
from collections import defaultdict
import math
from typing import List, Dict, Tuple
import hashlib # INTEGRATION: For robust hashing

class TextProcessor:
    """
    Handles text processing and conversion to vector representations.
    - Tokenization
    - Stop word removal
    - TF-IDF Vectorization with robust hashing
    """
    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim
        self.stopwords = set([
            "a", "an", "the", "in", "on", "of", "is", "for", "and", "to", "was",
            "it", "that", "with", "as", "by", "this", "are", "be", "at", "or",
            "from", "has", "i", "you", "he", "she", "we", "they", "not", "but"
        ])
        self._idf_counts = defaultdict(int)
        self._doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if word not in self.stopwords]

    def update_idf_counts(self, text_corpus: List[str]):
        print("📚 Updating IDF counts from corpus...")
        self._doc_count += len(text_corpus)
        for text in text_corpus:
            unique_tokens = set(self._tokenize(text))
            for token in unique_tokens:
                self._idf_counts[token] += 1
        print(f"✅ IDF updated. Total documents processed: {self._doc_count}")

    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        tf_counts = defaultdict(int)
        for token in tokens:
            tf_counts[token] += 1
        token_count = len(tokens)
        return {token: count / token_count for token, count in tf_counts.items()} if token_count > 0 else {}

    def _calculate_idf(self, token: str) -> float:
        if self._idf_counts.get(token, 0) == 0 or self._doc_count == 0:
            return 0.0
        return math.log(self.doc_count / (1 + self._idf_counts[token]))

    def text_to_vector(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """
        Converts text to a vector using TF-IDF and a robust hashing mechanism.
        """
        tokens = self._tokenize(text)
        tf = self._calculate_tf(tokens)
        tfidf_scores = {token: tf_val * self._calculate_idf(token) for token, tf_val in tf.items()}

        vector = [0.0] * self.vector_dim
        for token, score in tfidf_scores.items():
            # INTEGRATION: Use SHA-256 for a much better hash distribution
            hash_object = hashlib.sha256(token.encode())
            hex_dig = hash_object.hexdigest()
            hash_int = int(hex_dig, 16)
            index = hash_int % self.vector_dim
            vector[index] += score
        
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        sorted_keywords = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
        top_keywords = dict(sorted_keywords[:10])

        return vector, top_keywords
