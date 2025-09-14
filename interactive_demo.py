"""
Interactive Vector Space Model Demo
Run this to interactively query the wedding gown documents
"""

from vector_space_model import VectorSpaceModel

def interactive_demo():
    """
    Interactive demonstration of the VSM
    """
    print("=" * 60)
    print("INTERACTIVE VECTOR SPACE MODEL DEMO")
    print("=" * 60)
    
    # Initialize documents
    documents = [
        "User selected Wedding gown",
        "User ordered on-line rose flowers", 
        "User searched diamond ring",
        "User selected white wedding gown, online flowers, 3 carat diamond ring"
    ]
    
    print("Documents in the collection:")
    for i, doc in enumerate(documents, 1):
        print(f"d{i}: {doc}")
    print()
    
    # Create VSM
    vsm = VectorSpaceModel()
    vsm.add_documents(documents)
    
    print("VSM initialized! You can now query the documents.")
    print("Type 'quit' to exit, 'help' for commands, or enter a search query.")
    print()
    
    while True:
        try:
            query = input("Enter your query: ").strip()
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("- Enter any text to search documents")
                print("- 'vocab' - show vocabulary")
                print("- 'matrix' - show TF-IDF matrix")
                print("- 'similarity' - show similarity matrix")
                print("- 'quit' - exit")
                print()
                continue
            elif query.lower() == 'vocab':
                vsm.print_vocabulary()
                continue
            elif query.lower() == 'matrix':
                vsm.print_tfidf_matrix()
                continue
            elif query.lower() == 'similarity':
                similarity_matrix = vsm.get_document_similarity_matrix()
                doc_ids = [f"d{i+1}" for i in range(len(documents))]
                
                print("\nDocument Similarity Matrix:")
                print("-" * 50)
                header = "".ljust(8)
                for doc_id in doc_ids:
                    header += doc_id.ljust(10)
                print(header)
                print("-" * len(header))
                
                for i, doc_id in enumerate(doc_ids):
                    row = doc_id.ljust(8)
                    for j in range(len(doc_ids)):
                        row += f"{similarity_matrix[i, j]:.4f}".ljust(10)
                    print(row)
                print()
                continue
            elif not query:
                print("Please enter a query or command.")
                continue
            
            # Process query
            print(f"\nSearching for: '{query}'")
            results = vsm.query_documents(query)
            
            print("Ranked results:")
            print("-" * 30)
            for i, (doc_id, similarity) in enumerate(results, 1):
                doc_text = documents[int(doc_id[1]) - 1]
                print(f"{i}. {doc_id}: {similarity:.4f}")
                print(f"   Text: {doc_text}")
                print()
            
            if similarity > 0:
                print(f"Most relevant document: {results[0][0]} (similarity: {results[0][1]:.4f})")
            else:
                print("No relevant documents found.")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    interactive_demo()
