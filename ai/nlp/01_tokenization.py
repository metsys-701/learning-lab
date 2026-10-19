#=========================
#1. Imports
#=========================
# Import regex for pattern matching, Counter for frequency analysis, and typing for type hints
import re
from collections import Counter
from typing import List, Tuple

#=========================
#2. Tokenizer Class & Stop-Words Initialization
#=========================

class SimpleTokenizer:
    """
    A foundational Tokeizer class demonstrating text normalization,
    word-level tokenization, and stop-word filtering.
    
    """

    def __init__(self):
        # Initializing basic English stop-words to filter out
        #low-information words during the tokenization process
        self.stop_words =  {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "this", "that", "these", "those", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did", "to",
            "from", "in", "out", "on", "off", "over", "under", "again", "further"
        }

#=========================
#3. Text Normalization / Cleaning
#=========================
# Normalize text by lowercasing, removing punctuation, and stripping extra whitespaces

    def clean_text(self, text: str) -> str:
        text = text.lower()
        # Keep only alphanumeric characters and spaces, remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        text =re.sub(r"\s+", " ", text).strip()
        return text
    
#=========================
#4. Tokenization and Stop-Word Filtering
#=========================
# Split normalized text into tokens and filter out stop-words if requested

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        cleaned = self.clean_text(text)
        token = cleaned.split(" ")

        # Remove stop-words and any accidental empty strings
        if remove_stopwords:
            tokens = [token for token in token if token not in self.stop_words and token != ""]

        return tokens

#=========================
#5. Vocabulary Statistics
#=========================
# Calculate word frequency statistics using Counter

    def get_vocab_stats(self, tokens: List[str]) -> Tuple[int, int, List[Tuple[str, int]]]:
        word_counts = Counter(tokens)
        total_tokens = len(tokens)
        unique_token = len(word_counts)
        most_common = word_counts.most_common(5)
        return total_tokens, unique_token, most_common


#=========================
#6. Main Execution / Testing
#=========================
if __name__ == "__main__":
    sample_text = "Natural Language Processing is amazing! It helps computer understand, interstand, and manipulate human language."

    tokenizer = SimpleTokenizer()

    print("Original Text:")
    print(sample_text)
    print("-" * 40)

    tokens = tokenizer.tokenize(sample_text, remove_stopwords=True)
    print("Token (without stop-words):")
    print(tokens)
    print("-" * 40)

    total, unique, top_common = tokenizer.get_vocab_stats(tokens)
    print(f"Total Tokens: {total}")
    print(f"Unique Tokens: {unique}")
    print(f"Most Common Tokens: {top_common}")


#=========================
# Learning Notes & Lab Summary
#=========================
# - Tokenization is the foundational step of NLP where text is broken down into smaller units (words).
# - Text normalization (lowercasing, punctuation removal) ensures consistency across data.
# - Stop-words filtering removes high-frequency, low-information words to focus on meaningful content.
# - Using Python's built-in collections.Counter makes frequency analysis efficient and straightforward.