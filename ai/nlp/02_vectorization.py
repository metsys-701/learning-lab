#=========================
# 1. Imports
#=========================

from collections import Counter
from typing import List, Dict

#=========================
# 2. Vectorizer Class & Vocabulary Fitting
#=========================

class SimpleVectorizer:
    """

    A foudational Vectorizer class that converts tokenized documents
    intto numerical frequency vectors (Bag-of-Words model)
    
    """

    def __init__(self):
        # Initialize an empty dictionary to store the vocabulary (word -> index mapping)
        self.vocabulary: Dict[str, int] = {}

    def fit(self, documents: List[List[str]]):
        """
        Learns a vacobulary dictionary from all documents.
        
        """

        # 1. Collect all unique words across all documents, remove duplicates, and sort alphabetically
        unique_words = sorted(list(set(word for doc in documents for word in doc)))

        # 2. Assign a unique numerical ID to each word and store in the vocabulary dictionary
        self.vocabulary = {word: idx for idx, word in enumerate(unique_words)}
    def transform(self, document: List[str]) -> List[int]:
        """ 
        Transforms a single document into a numerical frequency vector based 
        on the learned vacobulary

        """
        # 1. Creat a zero-filled vector with the same lenght as the vacobulary 
        vector = [0] * len(self.vocabulary)

        # 2. Count the frequency of each word in the given document
        word_counts = Counter(document)


        # 3. If the word exists in our vocabulary, update its count at the correct
        for word, count in word_counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                vector[idx] = count
        return vector   

    def fit_transform(self, documents: List[List[str]]) -> List[List[int]]:
        """
        Fits the vocabulary on the documents and transforms them into vectors in one step.

        """

        # 1. Learn the vacobulary from all documents 
        self.fit(documents)


        # 2. Transform each document into its numerical vector representation
        return [self.transform(doc) for doc in documents]

#=========================
# Test Block
#=========================

if __name__ == "__main__":
    # Sample pre-processed tokenized documents (corpus)
    corpus = [
        ["natural", "language", "processing", "is", "amazing"],
        ["machine", "learning", "and", "natural", "language", "processing"],
        ["deep", "learning", "is", "amazing"]
    ]    

    print("--- Vectorization Test Starting ---")

    # Initialize the vectorizer
    vectorizer = SimpleVectorizer()

    # Fit and transfom the corpus 
    vectors = vectorizer.fit_transform(corpus)

    print("\nVocabulary Mapping:")
    print(vectorizer.vocabulary)

    print("\nResulting Vectors:")
    for i, vec in enumerate(vectors):
        print(f"Document {i+1}: {vec}")

#=========================
# Learning Notes & Lab Summary
#=========================
# - Vectorization converts text tokens into numerical vectors, enabling computers to process natural language mathematically.
# - The "fit" phase builds a global vocabulary dictionary, mapping every unique word to a unique numerical ID.
# - The "transform" phase applies the Bag-of-Words (BoW) model, counting word frequencies and placing them into their corresponding index positions.
# - Converting raw text into structured matrices bridges the gap between human language and machine learning algorithms.