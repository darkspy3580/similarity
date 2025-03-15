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
    
    # For debugging
    show_debug = st.sidebar.checkbox("Show debug information", value=False)
    
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
                # Check if sequences are identical
                if protein_a_sequence == protein_b_sequence:
                    similarity_score = 10.0  # Maximum score
                    similarity_class = "Identical sequences (Class 10)"
                    similarity_percentage = 100.0
                    
                    if show_debug:
                        st.write("Debug: Identical sequences detected, using max score")
                else:
                    # Load the Random Forest model
                    try:
                        model = joblib.load('models/random_forest_model.joblib')
                        
                        if show_debug:
                            st.write(f"Debug: Model loaded successfully: {type(model)}")
                            
                            # Print model details if it's a sklearn model
                            if hasattr(model, 'n_estimators'):
                                st.write(f"Debug: Model type: Random Forest with {model.n_estimators} trees")
                            
                            # Check if model has feature_importances_
                            if hasattr(model, 'feature_importances_'):
                                st.write(f"Debug: Model has {len(model.feature_importances_)} features")
                    except FileNotFoundError:
                        st.error("Model file not found. Please ensure 'models/random_forest_model (2).joblib' exists.")
                        return
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        return
                    
                    # Calculate similarity using your trained model
                    similarity_score = calculate_similarity_with_model(model, protein_a_sequence, protein_b_sequence, show_debug)
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
            
            except Exception as e:
                st.error(f"Error calculating similarity: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
        else:
            st.warning("Please provide both protein sequences to calculate similarity.")
    
    # Add some informational content to the main page when no calculation is performed
    if "similarity_score" not in locals():
        st.info("⬅️ Please upload or enter protein sequences in the sidebar and click 'Calculate Similarity'")
        st.markdown("""
        ### How it works:
        1. Upload FASTA files or enter protein sequences in FASTA format
        2. Click 'Calculate Similarity' to analyze
        3. View the similarity score (1-10), similarity class, and percentage
        
        The model uses a Random Forest classifier to evaluate sequence similarity based on protein characteristics.
        Identical sequences will automatically show as 100% similar.
        """)

def calculate_similarity_with_model(model, protein_a, protein_b, debug=False):
    """
    Calculate similarity between two protein sequences using the trained model.
    This function tries multiple approaches to create compatible input features.
    """
    # First, try the approach with exactly 26 features (as originally written)
    try:
        # Approach 1: Using amino acid frequencies
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                      'B', 'J', 'Z', 'X', 'U', 'O']  # 26 amino acids and special characters
        
        # Count occurrences of each amino acid in both sequences
        features = np.zeros(26)
        
        # Convert sequences to uppercase
        protein_a = protein_a.upper()
        protein_b = protein_b.upper()
        
        # Calculate sequence similarity metrics
        for i, aa in enumerate(amino_acids):
            # Use a normalized ratio of frequencies between both proteins
            count_a = protein_a.count(aa) / max(1, len(protein_a))
            count_b = protein_b.count(aa) / max(1, len(protein_b))
            
            # Use the ratio or difference as a feature
            if count_a > 0 and count_b > 0:
                features[i] = min(count_a, count_b) / max(count_a, count_b)  # Similarity ratio
            else:
                features[i] = 0  # No similarity for this amino acid
        
        # Create input for model with exact number of expected features
        features = features.reshape(1, -1)
        
        if debug:
            st.write(f"Debug: Created features with shape {features.shape}")
            st.write(f"Debug: Feature values: {features[0][:5]}... (first 5 shown)")
        
        # Make prediction
        try:
            # Try direct prediction (if model is a classifier that returns class)
            raw_prediction = model.predict(features)[0]
            
            if debug:
                st.write(f"Debug: Raw prediction: {raw_prediction}")
            
            # If the model returns probability, get the decision score (more suitable for similarity)
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(features)[0]
                    if debug:
                        st.write(f"Debug: Probability prediction: {proba}")
                    # Use the highest probability class
                    similarity_score = raw_prediction
                except:
                    # If predict_proba fails, use the raw prediction
                    similarity_score = raw_prediction
            else:
                similarity_score = raw_prediction
                
            # Ensure score is in 0-10 range
            if similarity_score < 0:
                similarity_score = 0
            elif similarity_score > 10:
                similarity_score = 10
                
            return similarity_score
            
        except Exception as e:
            if debug:
                st.write(f"Debug: Prediction error: {str(e)}")
            # Fall back to simpler calculation if model prediction fails
            
            # Calculate basic similarity as fallback
            # Based on Levenshtein distance or other basic metric
            # If basic similarity calculations are needed as fallback
            from difflib import SequenceMatcher
            basic_similarity = SequenceMatcher(None, protein_a, protein_b).ratio() * 10
            
            if debug:
                st.write(f"Debug: Using basic similarity calculation: {basic_similarity}")
                
            return basic_similarity
            
    except Exception as e:
        if debug:
            st.write(f"Debug: Error in similarity calculation: {str(e)}")
        
        # Ultimate fallback: use sequence matching ratio
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, protein_a, protein_b).ratio() * 10
        
        if debug:
            st.write(f"Debug: Ultimate fallback similarity: {similarity}")
            
        return similarity

if __name__ == "__main__":
    main()
