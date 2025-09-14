"""
Vector Space Model (VSM) Implementation for Wedding Gown Documents

This implementation demonstrates:
1. Document preprocessing and tokenization
2. Term Frequency (TF) calculation
3. Inverse Document Frequency (IDF) calculation
4. TF-IDF vector representation
5. Cosine similarity for document comparison
6. Query processing and document ranking
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class VectorSpaceModel:
    """
    A comprehensive Vector Space Model implementation for document retrieval
    """
    
    def __init__(self):
        self.documents = []
        self.vocabulary = set()
        self.tf_matrix = []
        self.idf_scores = {}
        self.tfidf_matrix = []
        self.document_vectors = {}
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by converting to lowercase, removing punctuation,
        and splitting into tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the VSM and build vocabulary
        """
        self.documents = documents
        processed_docs = []
        
        # Process each document
        for doc in documents:
            processed_tokens = self.preprocess_text(doc)
            processed_docs.append(processed_tokens)
            
            # Add tokens to vocabulary
            self.vocabulary.update(processed_tokens)
        
        # Convert vocabulary to sorted list for consistent indexing
        self.vocabulary = sorted(list(self.vocabulary))
        
        # Build TF matrix
        self._build_tf_matrix(processed_docs)
        
        # Calculate IDF scores
        self._calculate_idf_scores(processed_docs)
        
        # Build TF-IDF matrix
        self._build_tfidf_matrix()
        
        # Create document vectors
        self._create_document_vectors()
    
    def _build_tf_matrix(self, processed_docs: List[List[str]]):
        """
        Build Term Frequency matrix
        """
        self.tf_matrix = []
        
        for doc_tokens in processed_docs:
            # Count term frequencies in document
            term_counts = Counter(doc_tokens)
            
            # Create TF vector for this document
            tf_vector = []
            for term in self.vocabulary:
                tf_vector.append(term_counts.get(term, 0))
            
            self.tf_matrix.append(tf_vector)
    
    def _calculate_idf_scores(self, processed_docs: List[List[str]]):
        """
        Calculate Inverse Document Frequency scores
        """
        total_docs = len(processed_docs)
        
        for term in self.vocabulary:
            # Count documents containing this term
            doc_count = sum(1 for doc in processed_docs if term in doc)
            
            # Calculate IDF score
            if doc_count > 0:
                self.idf_scores[term] = math.log(total_docs / doc_count)
            else:
                self.idf_scores[term] = 0
    
    def _build_tfidf_matrix(self):
        """
        Build TF-IDF matrix by multiplying TF and IDF scores
        """
        self.tfidf_matrix = []
        
        for tf_vector in self.tf_matrix:
            tfidf_vector = []
            for i, tf_score in enumerate(tf_vector):
                term = self.vocabulary[i]
                idf_score = self.idf_scores[term]
                tfidf_score = tf_score * idf_score
                tfidf_vector.append(tfidf_score)
            
            self.tfidf_matrix.append(tfidf_vector)
    
    def _create_document_vectors(self):
        """
        Create normalized document vectors for similarity calculations
        """
        for i, tfidf_vector in enumerate(self.tfidf_matrix):
            # Calculate vector magnitude
            magnitude = math.sqrt(sum(score ** 2 for score in tfidf_vector))
            
            # Normalize vector
            if magnitude > 0:
                normalized_vector = [score / magnitude for score in tfidf_vector]
            else:
                normalized_vector = tfidf_vector
            
            self.document_vectors[f"d{i+1}"] = normalized_vector
    
    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
        magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def query_documents(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Process a query and return ranked documents
        """
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        
        # Create query vector
        query_vector = self._create_query_vector(query_tokens)
        
        # Calculate similarities with all documents
        similarities = []
        for doc_id, doc_vector in self.document_vectors.items():
            similarity = self.calculate_cosine_similarity(query_vector, doc_vector)
            similarities.append((doc_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        if top_k:
            return similarities[:top_k]
        return similarities
    
    def _create_query_vector(self, query_tokens: List[str]) -> List[float]:
        """
        Create TF-IDF vector for query
        """
        # Count term frequencies in query
        query_counts = Counter(query_tokens)
        
        # Create query vector
        query_vector = []
        for term in self.vocabulary:
            tf_score = query_counts.get(term, 0)
            idf_score = self.idf_scores.get(term, 0)
            tfidf_score = tf_score * idf_score
            query_vector.append(tfidf_score)
        
        # Normalize query vector
        magnitude = math.sqrt(sum(score ** 2 for score in query_vector))
        if magnitude > 0:
            query_vector = [score / magnitude for score in query_vector]
        
        return query_vector
    
    def get_document_similarity_matrix(self) -> np.ndarray:
        """
        Calculate similarity matrix between all document pairs
        """
        doc_ids = list(self.document_vectors.keys())
        n_docs = len(doc_ids)
        similarity_matrix = np.zeros((n_docs, n_docs))
        
        for i, doc1_id in enumerate(doc_ids):
            for j, doc2_id in enumerate(doc_ids):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity = self.calculate_cosine_similarity(
                        self.document_vectors[doc1_id],
                        self.document_vectors[doc2_id]
                    )
                    similarity_matrix[i][j] = similarity
        
        return similarity_matrix
    
    def print_vocabulary(self):
        """
        Print the vocabulary with term indices
        """
        print("Vocabulary:")
        print("-" * 50)
        for i, term in enumerate(self.vocabulary):
            print(f"{i:2d}: {term}")
        print()
    
    def print_tf_matrix(self):
        """
        Print the Term Frequency matrix
        """
        print("Term Frequency Matrix:")
        print("-" * 50)
        
        # Print header
        header = "Doc".ljust(8)
        for term in self.vocabulary:
            header += term.ljust(8)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for i, tf_vector in enumerate(self.tf_matrix):
            row = f"d{i+1}".ljust(8)
            for tf_score in tf_vector:
                row += f"{tf_score}".ljust(8)
            print(row)
        print()
    
    def print_idf_scores(self):
        """
        Print IDF scores for all terms
        """
        print("Inverse Document Frequency (IDF) Scores:")
        print("-" * 50)
        for term, idf_score in self.idf_scores.items():
            print(f"{term}: {idf_score:.4f}")
        print()
    
    def print_tfidf_matrix(self):
        """
        Print the TF-IDF matrix
        """
        print("TF-IDF Matrix:")
        print("-" * 50)
        
        # Print header
        header = "Doc".ljust(8)
        for term in self.vocabulary:
            header += term.ljust(10)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for i, tfidf_vector in enumerate(self.tfidf_matrix):
            row = f"d{i+1}".ljust(8)
            for tfidf_score in tfidf_vector:
                row += f"{tfidf_score:.4f}".ljust(10)
            print(row)
        print()
    
    def visualize_similarity_matrix(self, save_plot=True):
        """
        Create a heatmap visualization of document similarities
        """
        similarity_matrix = self.get_document_similarity_matrix()
        doc_ids = [f"d{i+1}" for i in range(len(self.documents))]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(similarity_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Cosine Similarity')
        
        # Set ticks and labels
        plt.xticks(range(len(doc_ids)), doc_ids)
        plt.yticks(range(len(doc_ids)), doc_ids)
        
        # Add text annotations
        for i in range(len(doc_ids)):
            for j in range(len(doc_ids)):
                plt.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                        ha='center', va='center', color='white' if similarity_matrix[i, j] < 0.5 else 'black')
        
        plt.title('Document Similarity Matrix')
        plt.xlabel('Documents')
        plt.ylabel('Documents')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('similarity_matrix.png', dpi=300, bbox_inches='tight')
            print("Similarity matrix plot saved as 'similarity_matrix.png'")
        else:
            plt.show()


def demonstrate_vsm():
    """
    Demonstrate the Vector Space Model with the wedding gown documents
    """
    print("=" * 80)
    print("VECTOR SPACE MODEL DEMONSTRATION")
    print("=" * 80)
    
    # Define the documents
    documents = [
        "User selected Wedding gown",
        "User ordered on-line rose flowers", 
        "User searched diamond ring",
        "User selected white wedding gown, online flowers, 3 carat diamond ring"
    ]
    
    print("Documents:")
    print("-" * 30)
    for i, doc in enumerate(documents, 1):
        print(f"d{i}: {doc}")
    print()
    
    # Create VSM instance
    vsm = VectorSpaceModel()
    vsm.add_documents(documents)
    
    # Print vocabulary
    vsm.print_vocabulary()
    
    # Print TF matrix
    vsm.print_tf_matrix()
    
    # Print IDF scores
    vsm.print_idf_scores()
    
    # Print TF-IDF matrix
    vsm.print_tfidf_matrix()
    
    # Demonstrate query processing
    print("Query Processing Examples:")
    print("-" * 50)
    
    queries = [
        "wedding gown",
        "diamond ring",
        "online flowers",
        "white wedding gown"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = vsm.query_documents(query)
        print("Ranked results:")
        for doc_id, similarity in results:
            print(f"  {doc_id}: {similarity:.4f}")
    
    # Print similarity matrix
    print("\nDocument Similarity Matrix:")
    print("-" * 50)
    similarity_matrix = vsm.get_document_similarity_matrix()
    doc_ids = [f"d{i+1}" for i in range(len(documents))]
    
    # Print header
    header = "".ljust(8)
    for doc_id in doc_ids:
        header += doc_id.ljust(10)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for i, doc_id in enumerate(doc_ids):
        row = doc_id.ljust(8)
        for j in range(len(doc_ids)):
            row += f"{similarity_matrix[i, j]:.4f}".ljust(10)
        print(row)
    
    # Create visualization
    print("\nGenerating similarity matrix visualization...")
    vsm.visualize_similarity_matrix()
    
    return vsm


def compare_with_sklearn():
    """
    Compare our implementation with scikit-learn's TfidfVectorizer
    """
    print("\n" + "=" * 80)
    print("COMPARISON WITH SCIKIT-LEARN")
    print("=" * 80)
    
    documents = [
        "User selected Wedding gown",
        "User ordered on-line rose flowers", 
        "User searched diamond ring",
        "User selected white wedding gown, online flowers, 3 carat diamond ring"
    ]
    
    # Our implementation
    vsm = VectorSpaceModel()
    vsm.add_documents(documents)
    
    # Scikit-learn implementation
    vectorizer = TfidfVectorizer()
    sklearn_tfidf = vectorizer.fit_transform(documents)
    
    print("Our Implementation TF-IDF Matrix:")
    print("-" * 50)
    vsm.print_tfidf_matrix()
    
    print("Scikit-learn TF-IDF Matrix:")
    print("-" * 50)
    sklearn_df = pd.DataFrame(
        sklearn_tfidf.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=[f"d{i+1}" for i in range(len(documents))]
    )
    print(sklearn_df.round(4))
    
    # Compare cosine similarities
    print("\nCosine Similarity Comparison:")
    print("-" * 50)
    
    # Our implementation
    our_similarity = vsm.get_document_similarity_matrix()
    
    # Scikit-learn implementation
    sklearn_similarity = cosine_similarity(sklearn_tfidf)
    
    print("Our Implementation:")
    print(our_similarity.round(4))
    
    print("\nScikit-learn Implementation:")
    print(sklearn_similarity.round(4))
    
    # Calculate difference
    difference = np.abs(our_similarity - sklearn_similarity)
    print(f"\nMaximum difference: {np.max(difference):.6f}")


if __name__ == "__main__":
    # Run the demonstration
    vsm = demonstrate_vsm()
    
    # Compare with scikit-learn
    compare_with_sklearn()
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print("""
    The Vector Space Model implementation demonstrates:
    
    1. Document Preprocessing: Text normalization and tokenization
    2. Term Frequency (TF): Count of terms in each document
    3. Inverse Document Frequency (IDF): Measures term rarity across documents
    4. TF-IDF: Combines TF and IDF to weight terms by importance
    5. Cosine Similarity: Measures angle between document vectors
    6. Query Processing: Ranks documents by relevance to query
    
    Key Insights:
    - Document d4 has the highest term diversity (most unique terms)
    - Documents d1 and d4 share "wedding gown" terms, showing high similarity
    - The model effectively captures semantic relationships between documents
    - TF-IDF weighting helps distinguish important terms from common ones
    """)
