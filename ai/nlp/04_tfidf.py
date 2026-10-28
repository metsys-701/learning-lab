#=========================
# 1. Imports
#=========================

import math
from typing import List, Dict


#=========================
# 2. TF-IDF Calculations
#=========================

def compute_term_frequency(document: List[str]) -> Dict[str, float]:
    """

    Computers Term Frequency (TF) for a single document.
    TF = (Count of term in document) / (Total terms in document)
    """

    term_counts = {}
    total_terms = len(document)

    if total_terms == 0:
        return {}

    for term in document:
        term_counts[term] = term_counts.get(term, 0) + 1
    # Normalize by total terms
    tf_dict = {term: count / total_terms for term, count in term_counts.items()}
    return tf_dict

def compute_inverse_document_frequency(corpus: List[List[str]]) -> Dict[str, float]:
    """
    Computes Inverse Document Frequency (IDF) for all terms in the corpus.
    IDF = log_e (Total Documents / Number of documents containing the term)
    """
    num_docs = len(corpus)
    idf_dict = {}

    # Get all unique terms across the corpus 
    all_terms = set(term for doc in corpus for term in doc)

    for term in all_terms:
        # Count how many documents contain this term
        docs_containing_term = sum(1 for doc in corpus if term in doc)
        
        # --- İŞTE EKSİK OLAN SATIR BURASI ---
        # Doğal logaritma ile IDF hesaplayıp sözlüğe ekliyoruz (+1 ile sıfıra bölünme/log(1)=0 durumunu güvenceye alabiliriz)
        idf_dict[term] = math.log(1.0 + (num_docs / docs_containing_term))

    return idf_dict

def compute_tfidf(corpus: List[List[str]]) -> List[Dict[str, float]]:
    """
    Computes TF-IDF for corpus of documents,
    TF-IDF _ TF * IDF
    """

    idf_values = compute_inverse_document_frequency(corpus)
    tfidf_corpus = []

    for doc in corpus:
        tf_values = compute_term_frequency(doc)
        tfidf_doc = {}
        for term, tf in tf_values.items():
            # TF-IDF score is TF multiplied by IDF
            tfidf_doc[term] = tf * idf_values.get(term, 0.0)
        tfidf_corpus.append(tfidf_doc)

    return tfidf_corpus

#=========================
# Test Block
#=========================

if __name__ == "__main__":
    # Sample preporcessed documents (corpus)
    corpus = [        
        ["yapay", "zeka", "ve", "dogal", "dil", "isleme"],
        ["yapay", "zeka", "gelecegi", "sekillendiriyor"],
        ["dogal", "dil", "isleme", "cok", "onemli"]
    ]

    print("-- TF-IDF Calculation Test Starting ---")

    tfidf_results = compute_tfidf(corpus)

    for i, doc_tfidf in enumerate(tfidf_results, 1):
        print(f"\nDocument {i} TF-IDF Scores:")
        for term, score in doc_tfidf.items():
            print(f"  -{term}: {score:.4f}")

#=========================
# Learning Notes & Lab Summary
#=========================
# - TF (Term Frequency) measures how frequently a term occurs in a document.
# - IDF (Inverse Document Frequency) measures how important a term is by penalizing overly common words across the corpus.
# - TF-IDF highlights domain-specific keywords while reducing the weight of generic terms.