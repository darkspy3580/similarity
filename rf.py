import streamlit as st
import numpy as np
import joblib  # Import joblib to load the random forest model
from sklearn.feature_extraction.text import CountVectorizer

# Function to read FASTA format
def read_fasta(file):
    fasta_dict = {}
    current_key = None
    current_sequence = []

    for line in file:
        line = line.strip()
        if line.startswith(">"):
            if current_key:
                fasta_dict[current_key] = "".join(current_sequence)
            current_key = line[1:]
            current_sequence = []
        else:
            current_sequence.append(line)

    if current_key:
        fasta_dict[current_key] = "".join(current_sequence)

    return fasta_dict

# Define the main function
def main():
    st.title("Protein Similarity Using ML")

    # Sidebar for uploading protein sequences
    st.sidebar.title("Upload Protein Sequences")
    protein_a_file = st.sidebar.file_uploader("Upload FASTA file for Protein A")
    protein_b_file = st.sidebar.file_uploader("Upload FASTA file for Protein B")
    protein_a_input = st.sidebar.text_area("Enter Protein A sequence (FASTA format)")
    protein_b_input = st.sidebar.text_area("Enter Protein B sequence (FASTA format)")

    # Submit button
    if st.sidebar.button("Calculate Similarity"):
        # Read sequences from file or text input
        if protein_a_file:
            protein_a_content = protein_a_file.getvalue().decode("utf-8")
            protein_a_sequence = read_fasta(protein_a_content.splitlines())
            protein_a_sequence = list(protein_a_sequence.values())[0]
        elif protein_a_input:
            protein_a_sequence = protein_a_input.splitlines()[-1]  # Get the last line as the sequence

        if protein_b_file:
            protein_b_content = protein_b_file.getvalue().decode("utf-8")
            protein_b_sequence = read_fasta(protein_b_content.splitlines())
            protein_b_sequence = list(protein_b_sequence.values())[0]
        elif protein_b_input:
            protein_b_sequence = protein_b_input.splitlines()[-1]  # Get the last line as the sequence

        # Perform similarity calculation using random forest model
        similarity_score = calculate_similarity_random_forest(protein_a_sequence, protein_b_sequence)

        # Display results
        st.success(f"Similarity Score: {similarity_score:.2f}")

# Actual similarity calculation function using Random Forest model
def calculate_similarity_random_forest(protein_a_sequence, protein_b_sequence):
    # Load the Random Forest model from the 'models' subfolder
    model = joblib.load('models/random_forest_model.joblib')

    # Prepare data for model input
    sequences = [protein_a_sequence, protein_b_sequence]

    # Vectorize the sequences (adjust max_features or other parameters as needed)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=26)  # Ensure this matches the training setup
    X = vectorizer.fit_transform(sequences)

    # Ensure we only use the first sequence's features for prediction
    if X.shape[1] != 26:
        raise ValueError(f"Expected 26 features, but got {X.shape[1]} features.")

    # Predict similarity using the model
    similarity_score = model.predict(X)[0]  # Get similarity score

    # Scale the score from 0-10 to 0-100
    scaled_similarity_score = similarity_score * 10  # Scale to 0-100

    return scaled_similarity_score

# Run the app
if __name__ == "__main__":
    main()
