import streamlit as st
import numpy as np
import joblib
from difflib import SequenceMatcher
import re

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

def calculate_sequence_similarity(seq1, seq2):
    """Calculate sequence similarity score using multiple methods and return a score from 0-10"""
    # Method 1: Use SequenceMatcher ratio (accounts for sequence order and content)
    ratio_score = SequenceMatcher(None, seq1, seq2).ratio() * 10
    
    # Method 2: Calculate amino acid composition similarity
    # Count amino acids in both sequences
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Get counts for each sequence
    counts_1 = {aa: seq1.upper().count(aa) / max(1, len(seq1)) for aa in amino_acids}
    counts_2 = {aa: seq2.upper().count(aa) / max(1, len(seq2)) for aa in amino_acids}
    
    # Calculate composition similarity
    composition_similarity = 0
    for aa in amino_acids:
        composition_similarity += (1 - abs(counts_1[aa] - counts_2[aa])) / len(amino_acids)
    
    composition_score = composition_similarity * 10
    
    # Method 3: Calculate k-mer similarity (shared subsequences)
    k = 3  # length of k-mers
    kmers_1 = set(seq1[i:i+k] for i in range(len(seq1)-k+1))
    kmers_2 = set(seq2[i:i+k] for i in range(len(seq2)-k+1))
    
    if not kmers_1 or not kmers_2:
        kmer_score = 0
    else:
        shared = len(kmers_1.intersection(kmers_2))
        total = len(kmers_1.union(kmers_2))
        kmer_score = (shared / total) * 10
    
    # Combine scores with appropriate weights
    # Give more weight to ratio_score as it considers sequence order
    final_score = 0.4 * ratio_score + 0.3 * composition_score + 0.3 * kmer_score
    
    return min(10.0, max(0.0, final_score))  # Clamp between 0 and 10

def extract_features_from_sequences(seq1, seq2):
    """Extract features from protein sequences for the random forest model.
    Returns a feature vector that the model can use for prediction."""
    
    # Feature 1: Sequence similarity using SequenceMatcher
    ratio_score = SequenceMatcher(None, seq1, seq2).ratio()
    
    # Feature 2: Length ratio
    len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
    
    # Feature 3: Amino acid composition similarity
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    counts_1 = {aa: seq1.upper().count(aa) / max(1, len(seq1)) for aa in amino_acids}
    counts_2 = {aa: seq2.upper().count(aa) / max(1, len(seq2)) for aa in amino_acids}
    
    composition_diffs = []
    for aa in amino_acids:
        composition_diffs.append(abs(counts_1[aa] - counts_2[aa]))
    
    composition_similarity = 1 - (sum(composition_diffs) / len(amino_acids))
    
    # Feature 4: K-mer similarity
    k = 3  # length of k-mers
    kmers_1 = set(seq1[i:i+k] for i in range(len(seq1)-k+1) if i+k <= len(seq1))
    kmers_2 = set(seq2[i:i+k] for i in range(len(seq2)-k+1) if i+k <= len(seq2))
    
    if not kmers_1 or not kmers_2:
        kmer_similarity = 0
    else:
        shared = len(kmers_1.intersection(kmers_2))
        total = len(kmers_1.union(kmers_2))
        kmer_similarity = shared / total
    
    # Feature 5: Calculate molecular weight difference (approximation)
    # Molecular weights in Daltons (approximate)
    mw = {'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2, 
          'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2, 
          'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2, 
          'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2}
    
    mw1 = sum(mw.get(aa.upper(), 0) for aa in seq1)
    mw2 = sum(mw.get(aa.upper(), 0) for aa in seq2)
    mw_ratio = min(mw1, mw2) / max(mw1, mw2)
    
    # Feature 6: Longest common subsequence ratio
    def lcs_length(s1, s2):
        m, n = len(s1), len(s2)
        # Use shorter strings if sequences are too long
        if m > 1000 or n > 1000:
            s1 = s1[:1000]
            s2 = s2[:1000]
            m, n = len(s1), len(s2)
            
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # Only compute LCS for reasonably sized sequences to avoid performance issues
    if len(seq1) < 2000 and len(seq2) < 2000:
        lcs_len = lcs_length(seq1, seq2)
        lcs_ratio = lcs_len / max(len(seq1), len(seq2))
    else:
        # For very long sequences, estimate using the first 2000 characters
        lcs_len = lcs_length(seq1[:2000], seq2[:2000])
        lcs_ratio = lcs_len / 2000
    
    # Combine all features into a single vector
    features = [
        ratio_score,                # Sequence matcher similarity
        len_ratio,                  # Length ratio
        composition_similarity,     # AA composition similarity
        kmer_similarity,            # K-mer similarity
        mw_ratio,                   # Molecular weight ratio
        lcs_ratio                   # Longest common subsequence ratio
    ]
    
    # Add individual amino acid frequency differences
    features.extend(composition_diffs)
    
    return features

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
    
    # Analysis method selection
    analysis_method = st.sidebar.radio(
        "Choose Analysis Method",
        ["Sequence-based Similarity", "Random Forest Model"]
    )
    
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
        
        # Clean sequences (remove whitespace and non-amino acid characters)
        if protein_a_sequence:
            protein_a_sequence = re.sub(r'[^A-Za-z]', '', protein_a_sequence)
        if protein_b_sequence:
            protein_b_sequence = re.sub(r'[^A-Za-z]', '', protein_b_sequence)
        
        # Validate sequences and calculate similarity
        if protein_a_sequence and protein_b_sequence:
            try:
                # Check if sequences are identical
                if protein_a_sequence == protein_b_sequence:
                    similarity_score = 10.0  # Maximum score
                    similarity_class = "Identical sequences (Class 10)"
                    similarity_percentage = 100.0
                else:
                    if analysis_method == "Sequence-based Similarity":
                        # Use the sequence-based similarity calculation
                        similarity_score = calculate_sequence_similarity(protein_a_sequence, protein_b_sequence)
                        st.info("Using sequence-based similarity calculation.")
                    else:
                        # Try to use the random forest model
                        try:
                            # Load the model
                            model = joblib.load('models/random_forest_model.joblib')
                            
                            # Extract features from sequences for the model
                            features = extract_features_from_sequences(protein_a_sequence, protein_b_sequence)
                            
                            # Use the model to predict similarity
                            model_score = model.predict([features])[0]
                            
                            st.info(f"Random Forest Model raw prediction: {model_score:.4f}")
                            
                            if model_score <= 0:
                                st.warning("Model prediction was 0 or negative. Using sequence-based similarity as fallback.")
                                similarity_score = calculate_sequence_similarity(protein_a_sequence, protein_b_sequence)
                            else:
                                similarity_score = min(10.0, max(0.0, model_score))  # Ensure score is in range [0, 10]
                                st.success("Successfully used Random Forest model for prediction.")
                                
                        except Exception as e:
                            st.warning(f"Model could not be loaded or error in prediction: {str(e)}. Using sequence-based similarity instead.")
                            similarity_score = calculate_sequence_similarity(protein_a_sequence, protein_b_sequence)
                    
                    similarity_class = get_similarity_class(similarity_score)
                    # Convert score (1-10) to percentage (0-100%)
                    similarity_percentage = (similarity_score / 10) * 100
                
                # Display results
                st.success(f"Similarity Score: {similarity_score:.2f}/10")
                st.info(f"Similarity Class: {similarity_class}")
                st.info(f"Similarity Percentage: {similarity_percentage:.1f}%")
                
                # Create a visual indicator of similarity (progress bar)
                st.progress(similarity_percentage/100)
                
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


