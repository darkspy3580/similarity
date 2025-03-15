import streamlit as st
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Function to read FASTA format
def read_fasta(lines):
    fasta_dict = {}
    current_key = None
    current_sequence = []
    
    for line in lines:
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

def extract_features(protein_a, protein_b):
    """Extract features from the two protein sequences for model input"""
    # Create combined sequence for comparison
    combined_seq = protein_a + "_" + protein_b
    
    # Use the same vectorizer configuration as used during training
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
    X = vectorizer.fit_transform([combined_seq])
    
    return X

def get_similarity_class(score):
    """Convert numerical score to similarity class description"""
    if score <= 1:
        return "No similarity (Class 1)"
    elif score <= 2:
        return "Very low similarity (Class 2)"
    elif score <= 3:
        return "Low similarity (Class 3)"
    elif score <= 4:
        return "Low-moderate similarity (Class 4)"
    elif score <= 5:
        return "Moderate similarity (Class 5)"
    elif score <= 6:
        return "Moderate-high similarity (Class 6)"
    elif score <= 7:
        return "High similarity (Class 7)"
    elif score <= 8:
        return "Very high similarity (Class 8)"
    elif score <= 9:
        return "Extremely high similarity (Class 9)"
    else:
        return "Near identical (Class 10)"

def main():
    st.title("Protein Similarity Using ML")
    st.write("Upload or enter two protein sequences to calculate their similarity score and class.")
    
    # Sidebar for uploading protein sequences
    st.sidebar.title("Upload Protein Sequences")
    protein_a_file = st.sidebar.file_uploader("Upload FASTA file for Protein A", type=["fasta", "fa", "txt"])
    protein_b_file = st.sidebar.file_uploader("Upload FASTA file for Protein B", type=["fasta", "fa", "txt"])
    
    # Text input alternatives
    st.sidebar.markdown("### Or enter sequences directly:")
    protein_a_input = st.sidebar.text_area("Enter Protein A sequence (FASTA format)")
    protein_b_input = st.sidebar.text_area("Enter Protein B sequence (FASTA format)")
    
    # Submit button
    if st.sidebar.button("Calculate Similarity"):
        # Process and validate inputs
        protein_a_sequence = None
        protein_b_sequence = None
        
        # Get protein A sequence
        if protein_a_file is not None:
            protein_a_content = protein_a_file.getvalue().decode("utf-8")
            protein_a_dict = read_fasta(protein_a_content.splitlines())
            if protein_a_dict:
                protein_a_sequence = list(protein_a_dict.values())[0]
            else:
                st.error("Invalid FASTA format for Protein A")
        elif protein_a_input:
            try:
                if protein_a_input.startswith(">"):
                    protein_a_dict = read_fasta(protein_a_input.splitlines())
                    protein_a_sequence = list(protein_a_dict.values())[0]
                else:
                    # Assume raw sequence without header
                    protein_a_sequence = protein_a_input.strip()
            except Exception as e:
                st.error(f"Error processing Protein A input: {str(e)}")
        
        # Get protein B sequence
        if protein_b_file is not None:
            protein_b_content = protein_b_file.getvalue().decode("utf-8")
            protein_b_dict = read_fasta(protein_b_content.splitlines())
            if protein_b_dict:
                protein_b_sequence = list(protein_b_dict.values())[0]
            else:
                st.error("Invalid FASTA format for Protein B")
        elif protein_b_input:
            try:
                if protein_b_input.startswith(">"):
                    protein_b_dict = read_fasta(protein_b_input.splitlines())
                    protein_b_sequence = list(protein_b_dict.values())[0]
                else:
                    # Assume raw sequence without header
                    protein_b_sequence = protein_b_input.strip()
            except Exception as e:
                st.error(f"Error processing Protein B input: {str(e)}")
        
        # Validate sequences and calculate similarity
        if protein_a_sequence and protein_b_sequence:
            try:
                # Load the Random Forest model
                try:
                    model = joblib.load('models/random_forest_model (2).joblib')
                except FileNotFoundError:
                    st.error("Model file not found. Please ensure 'models/random_forest_model.joblib' exists.")
                    return
                
                # Calculate similarity
                similarity_score = calculate_similarity(model, protein_a_sequence, protein_b_sequence)
                similarity_class = get_similarity_class(similarity_score)
                
                # Display results
                st.success(f"Similarity Score: {similarity_score:.2f}/10")
                st.info(f"Similarity Class: {similarity_class}")
                
                # Show protein information
                st.subheader("Protein Sequences")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Protein A:**")
                    st.text_area("Sequence A", protein_a_sequence[:100] + "..." if len(protein_a_sequence) > 100 else protein_a_sequence, height=100, disabled=True)
                    st.write(f"Length: {len(protein_a_sequence)} amino acids")
                with col2:
                    st.write("**Protein B:**")
                    st.text_area("Sequence B", protein_b_sequence[:100] + "..." if len(protein_b_sequence) > 100 else protein_b_sequence, height=100, disabled=True)
                    st.write(f"Length: {len(protein_b_sequence)} amino acids")
            
            except Exception as e:
                st.error(f"Error calculating similarity: {str(e)}")
        else:
            st.warning("Please provide both protein sequences to calculate similarity.")
    
    # Add some informational content to the main page when no calculation is performed
    if "similarity_score" not in locals():
        st.info("⬅️ Please upload or enter protein sequences in the sidebar and click 'Calculate Similarity'")
        st.markdown("""
        ### How it works:
        1. Upload FASTA files or enter protein sequences in FASTA format
        2. Click 'Calculate Similarity' to analyze
        3. View the similarity score (1-10) and corresponding similarity class
        
        The model uses a Random Forest classifier to evaluate sequence similarity based on character n-grams.
        """)

def calculate_similarity(model, protein_a, protein_b):
    """Calculate similarity between two protein sequences using the random forest model"""
    # Prepare data for model input
    try:
        # Create feature vectors for the model
        # Method 1: Using the approach that matches your model's training
        sequences = [protein_a, protein_b]
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
        X = vectorizer.fit_transform(sequences)
        
        # Extract features from both sequences
        # You may need to adjust this depending on how your model was trained
        features = np.zeros(X.shape[1] * 2)  # Features for both sequences
        features[:X.shape[1]] = X[0].toarray()[0]  # First sequence features
        features[X.shape[1]:] = X[1].toarray()[0]  # Second sequence features
        
        # Reshape for model input
        features = features.reshape(1, -1)
        
        # Predict similarity using the model
        similarity_score = model.predict(features)[0]
        
        return similarity_score
    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        st.info("Try alternative feature extraction method...")
        
        # Method 2: Alternative simpler approach if the above fails
        try:
            # Create combined sequence and extract features
            combined = protein_a + "_SEP_" + protein_b
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=100)
            X = vectorizer.fit_transform([combined])
            
            # Predict using the model
            similarity_score = model.predict(X)[0]
            return similarity_score
        except Exception as e2:
            raise Exception(f"Both feature extraction methods failed: {str(e2)}")

if __name__ == "__main__":
    main()
