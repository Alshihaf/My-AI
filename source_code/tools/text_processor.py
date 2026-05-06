
"""
Text Processor - From Words to Vectors

This module provides the essential bridge between textual information read from
files and the abstract vector representations used by the CognitiveEngine and
the SamanticGarden. It starts with a simple TF-IDF implementation.
"""
import re
from collections import defaultdict
import math
from typing import List, Dict, Tuple

class TextProcessor:
    """
    Handles text processing and conversion to vector representations.
    - Tokenization
    - Stop word removal
    - TF-IDF Vectorization
    """
    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim
        self.stopwords = set([
            "a", "an", "the", "in", "on", "of", "is", "for", "and", "to", "was",
            "it", "that", "with", "as", "by", "this", "are", "be", "at", "or",
            "from", "has", "i", "you", "he", "she", "we", "they", "not", "but"
            # More can be added
        ])
        self._idf_counts = defaultdict(int)
        self._doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple text tokenization: lowercase, alphanumeric, remove stopwords."""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if word not in self.stopwords]

    def update_idf_counts(self, text_corpus: List[str]):
        """
        First pass over a corpus to calculate Inverse Document Frequency (IDF) scores.
        This should be done periodically on a collection of documents to keep IDF relevant.
        """
        print("📚 Updating IDF counts from corpus...")
        self._doc_count += len(text_corpus)
        for text in text_corpus:
            unique_tokens = set(self._tokenize(text))
            for token in unique_tokens:
                self._idf_counts[token] += 1
        print(f"✅ IDF updated. Total documents processed: {self._doc_count}")

    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculates Term Frequency for tokens in a single document."""
        tf_counts = defaultdict(int)
        for token in tokens:
            tf_counts[token] += 1
        
        token_count = len(tokens)
        if token_count == 0:
            return {}
            
        tf = {token: count / token_count for token, count in tf_counts.items()}
        return tf

    def _calculate_idf(self, token: str) -> float:
        """Calculates Inverse Document Frequency for a single token."""
        if self._idf_counts[token] == 0 or self._doc_count == 0:
            return 0.0 # Return 0 for unseen words to avoid division by zero
        return math.log(self._doc_count / (1 + self._idf_counts[token]))

    def text_to_vector(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """
        Converts a single piece of text into a fixed-dimension vector using TF-IDF.
        Also returns the most important keywords based on their TF-IDF scores.
        """
        tokens = self._tokenize(text)
        tf = self._calculate_tf(tokens)

        tfidf_scores = {token: tf_val * self._calculate_idf(token) for token, tf_val in tf.items()}

        # Create a vector representation
        vector = [0.0] * self.vector_dim
        for token, score in tfidf_scores.items():
            # Use a simple hash to project the token's score onto the vector dimensions
            # This is a basic way to get a consistent vector representation.
            index = hash(token) % self.vector_dim
            vector[index] += score
        
        # Normalize the vector to have a unit length (L2 norm)
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        # Get top keywords
        sorted_keywords = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
        top_keywords = dict(sorted_keywords[:10])

        return vector, top_keywords

# Example usage (for testing)
if __name__ == '__main__':
    corpus = [
        "The agent must explore the environment to find new information.",
        "Learning involves processing information and storing it as knowledge.",
        "The samantic garden grows as the agent learns new concepts.",
        "Consolidation is the process of strengthening and pruning memories."
    ]
    processor = TextProcessor(vector_dim=64)

    # 1. Build IDF from a corpus
    processor.update_idf_counts(corpus)

    # 2. Convert a new document to a vector
    new_text = "The agent is learning about the samantic garden and exploring information."
    text_vector, keywords = processor.text_to_vector(new_text)

    print(f"\n--- Analysis of: '{new_text}' ---")
    print(f"Top Keywords: {keywords}")
    print(f"Generated Vector (first 10 dims): {text_vector[:10]}")
    print(f"Vector Dimension: {len(text_vector)}")

