"""
Simple demonstration script for the Vector Space Model
Run this to see the VSM in action with the wedding gown documents
"""

from vector_space_model import demonstrate_vsm, compare_with_sklearn

if __name__ == "__main__":
    print("Starting Vector Space Model Demonstration...")
    print("This will show how VSM works with the wedding gown documents.\n")
    
    # Run the main demonstration
    vsm = demonstrate_vsm()
    
    # Compare with scikit-learn for validation
    compare_with_sklearn()
    
    print("\nDemonstration complete!")
    print("Check the generated plots for visual analysis.")
