import streamlit as st
import joblib  # Import joblib to load the random forest model
from sklearn.feature_extraction.text import CountVectorizer

# Function to map similarity scores to classes
def map_values(value):
    if 1 <= value <= 10:
        return 0
    elif 11 <= value <= 20:
        return 1
    elif 21 <= value <= 30:
        return 2
    elif 31 <= value <= 40:
        return 3
    elif 41 <= value <= 50:
        return 4
    elif 51 <= value <= 60:
        return 5
    elif 61 <= value <= 70:
        return 6
    elif 71 <= value <= 80:
        return 7
    elif 81 <= value <= 90:
        return 8
    elif 91 <= value <= 100:
        return 9
    else:
        return -1  # Default case (should not occur)

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

# Function to calculate similarity using Random Forest model
def calculate_similarity_class(protein_a_sequence, protein_b_sequence):
    # If both sequences are identical, return 100% similarity
    if protein_a_sequence == protein_b_sequence:
        return "100% Similar"

    # Load the Random Forest model
    model = joblib.load('models/random_forest_model.joblib')

    # Prepare data for model input
    sequences = [protein_a_sequence, protein_b_sequence]

    # Vectorize the sequences
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=26)
    X = vectorizer.fit_transform(sequences)

    # Ensure we only use the first sequence's features for prediction
    if X.shape[1] != 26:
        raise ValueError(f"Expected 26 features, but got {X.shape[1]} features.")

    # Predict similarity using the model
    similarity_score = model.predict(X)[0]  # Get similarity score

    # Scale the score from 0-10 to 0-100
    scaled_similarity_score = similarity_score * 10  

    # Map to similarity class
    similarity_class = map_values(scaled_similarity_score)

    return f"Class: {similarity_class}"

# Define the main function
def main():
    st.title("Protein Similarity Prediction")

    # Sidebar for uploading protein sequences
    st.sidebar.title("Upload Protein Sequences")
    protein_a_file = st.sidebar.file_uploader("Upload FASTA file for Protein A")
    protein_b_file = st.sidebar.file_uploader("Upload FASTA file for Protein B")
    protein_a_input = st.sidebar.text_area("Enter Protein A sequence (FASTA format)")
    protein_b_input = st.sidebar.text_area("Enter Protein B sequence (FASTA format)")

    # Submit button
    if st.sidebar.button("Calculate Similarity"):
        # Read sequences from file or text input
        protein_a_sequence = None
        protein_b_sequence = None

        if protein_a_file:
            protein_a_content = protein_a_file.getvalue().decode("utf-8")
            protein_a_sequence = read_fasta(protein_a_content.splitlines())
            protein_a_sequence = list(protein_a_sequence.values())[0]
        elif protein_a_input:
            protein_a_sequence = protein_a_input.splitlines()[-1]  # Get last line as sequence

        if protein_b_file:
            protein_b_content = protein_b_file.getvalue().decode("utf-8")
            protein_b_sequence = read_fasta(protein_b_content.splitlines())
            protein_b_sequence = list(protein_b_sequence.values())[0]
        elif protein_b_input:
            protein_b_sequence = protein_b_input.splitlines()[-1]  # Get last line as sequence

        # Validate input
        if not protein_a_sequence or not protein_b_sequence:
            st.error("Please provide valid protein sequences for both inputs.")
            return

        # Perform similarity calculation
        similarity_result = calculate_similarity_class(protein_a_sequence, protein_b_sequence)

        # Display results
        st.success(f"Similarity Result: {similarity_result}")

if __name__ == "__main__":
    main()
