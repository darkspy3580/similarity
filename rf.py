import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    st.title("Protein Similarity Using CNN Model")

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

        # Perform similarity calculation using CNN model
        similarity_score = calculate_similarity_cnn(protein_a_sequence, protein_b_sequence)

        # Display results
        st.success(f"Similarity Score: {similarity_score:.2f}")

# Actual similarity calculation function using CNN model
def calculate_similarity_cnn(protein_a_sequence, protein_b_sequence):
    # Load the CNN model
    model = load_model('/models/cnn_model.h5')

    # Tokenize and pad sequences
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts([protein_a_sequence, protein_b_sequence])
    tokenized_sequences = tokenizer.texts_to_sequences([protein_a_sequence, protein_b_sequence])
    padded_sequences = pad_sequences(tokenized_sequences, maxlen=26, padding='post')

    # Predict similarity using the model
    similarity_scores = model.predict(padded_sequences.reshape(-1, 26, 1))
    similarity_score = similarity_scores[0][0]  # Get similarity score for protein A
    return similarity_score

if __name__ == "__main__":
    main()
