#=========================
# 1. Imports
#=========================

import math
from typing import List

#=========================
# 2. Similarity Functions & Calculations
#=========================

def dot_product(vec1: List[int], vec2: List[int]) -> float:
    """
    Calculates the dot product of two numerical vectors.
    Formula: sum(a_i * b_i)
    """ 
    return sum(a * b for a, b in zip(vec1, vec2))

def vector_magnitude(vec: List[int]) -> float:
    """
    Calculates the magnitude (Euclidean length) of a vector.
    Formula: sqrt(sum(a_i^2))
    """
    return math.sqrt(sum(a ** 2 for a in vec))

def cosine_similarity(vec1: List[int], vec2: List[int]) -> float:
    """
    Calculates the cosine similarity between two numerical vectors.
    Formula: (A . B) / (||A|| * ||B||)
    """
    dot = dot_product(vec1, vec2)
    mag1 = vector_magnitude(vec1)
    mag2 = vector_magnitude(vec2)

    # Edge case: avoid division by zero if a vector is empty or all zeros
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    return dot / (mag1 * mag2)

#=========================
# Test Block
#=========================

if __name__ == "__main__":
    # Sample document vectors (e.g., from our Bag-of-Words model)
    doc_a = [1, 2, 1, 0, 0]
    doc_b = [1, 2, 1, 0, 0]  # Identical to Doc A -> similarity should be 1.0
    doc_c = [0, 0, 0, 3, 2]  # Completely different -> similarity should be 0.0
    doc_d = [1, 1, 0, 0, 0]  # Partially similar

    print("--- Cosine Similarity Test Starting ---")

    sim_ab = cosine_similarity(doc_a, doc_b)
    sim_ac = cosine_similarity(doc_b, doc_c)
    sim_ad = cosine_similarity(doc_a, doc_d)

    print(f"\nSimilarity between Doc A and Doc B: {sim_ab:.4f}")
    print(f"Similarity between Doc A and Doc C: {sim_ac:.4f}")
    print(f"Similarity between Doc A and Doc D: {sim_ad:.4f}")

#=========================
# Learning Notes & Lab Summary
#=========================
# - Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space.
# - It is independent of document length, focusing purely on orientation and word proportions.
# - The dot product calculates the overlapping word frequencies, while vector magnitudes normalize the score between 0.0 and 1.0.