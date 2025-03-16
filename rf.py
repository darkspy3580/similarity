import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
from io import StringIO

# Function to read FASTA format
def read_fasta(text):
    fasta_dict = {}
    current_key = None
    current_sequence = []
    
    for line in text.splitlines():
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

# Function to extract protein features using peptide library methods
def extract_features_from_sequence(sequence):
    """Extract protein features for RF model prediction"""
    features = {}
    
    # Clean sequence to ensure valid amino acids
    valid_aa = "ACDEFGHIKLMNPQRSTVWY"
    sequence = ''.join(aa for aa in sequence.upper() if aa in valid_aa)
    
    # Skip if sequence is too short
    if len(sequence) < 5:
        return None
    
    try:
        # Initialize Bio.SeqUtils.ProtParam for analysis
        prot = ProteinAnalysis(sequence)
        
        # Extract features
        features = {
            'AlipIndex': prot.aliphatic_index(),
            'Autocorr': np.mean(prot.protein_scale(window=7, param_dict='Flex')),
            'Autocov': np.var(prot.protein_scale(window=7, param_dict='Flex')),
            'boman': sum(prot.protein_scale(window=7, param_dict='hw')),
            'charge': prot.charge_at_pH(7.0),
            'crosscov': 0,  # Placeholder, calculate if needed
            'hydrmom': prot.gravy(),
            'isoelctricp': prot.isoelectric_point(),
            'instablility': prot.instability_index(),
            'massshift': sum(prot.protein_scale(window=7, param_dict='Flex')),
            'molweight': prot.molecular_weight(),
            'mz': prot.molecular_weight() / abs(prot.charge_at_pH(7.0)) if prot.charge_at_pH(7.0) != 0 else 0,
            'sc': len([aa for aa in sequence if aa in 'DEHKR']) / max(1, len(sequence)),
            'length': len(sequence)
        }
        
        return features
    
    except Exception as e:
        st.error(f"Error extracting protein features: {str(e)}")
        return None

# Function to create feature dataframe from protein A and B
def create_feature_dataframe(p1_features, p2_features):
    """Combine protein A and B features into a single dataframe row"""
    
    # Create combined features dictionary
    features = {}
    
    # Add P1 features
    for key, value in p1_features.items():
        features[f'P1_{key}' if key != 'length' else 'P1'] = value
        
    # Add P2 features
    for key, value in p2_features.items():
        features[f'P2_{key}' if key != 'length' else 'P2'] = value
    
    # Create dataframe with single row
    df = pd.DataFrame([features])
    
    return df

# Function to get class description based on predicted value
def get_similarity_class(value):
    """Convert numerical prediction to similarity class"""
    if 1 <= value <= 10:
        return 0, "Very Low Similarity (1-10%)"
    elif 11 <= value <= 20:
        return 1, "Low Similarity (11-20%)"
    elif 21 <= value <= 30:
        return 2, "Low-Moderate Similarity (21-30%)"
    elif 31 <= value <= 40:
        return 3, "Moderate Similarity (31-40%)"
    elif 41 <= value <= 50:
        return 4, "Moderate Similarity (41-50%)"
    elif 51 <= value <= 60:
        return 5, "Moderate-High Similarity (51-60%)"
    elif 61 <= value <= 70:
        return 6, "High Similarity (61-70%)"
    elif 71 <= value <= 80:
        return 7, "High Similarity (71-80%)"
    elif 81 <= value <= 90:
        return 8, "Very High Similarity (81-90%)"
    elif 91 <= value <= 100:
        return 9, "Extremely High Similarity (91-100%)"
    else:
        return None, "Unknown"

def main():
    st.title("Protein Similarity Class Prediction")
    st.write("Upload or enter two protein sequences to predict their similarity class.")
    
    # Sidebar for uploading protein sequences
    st.sidebar.title("Input Protein Sequences")
    
    input_method = st.sidebar.radio(
        "Choose Input Method",
        ["Upload FASTA Files", "Enter FASTA Text"]
    )
    
    protein_a_sequence = None
    protein_b_sequence = None
    
    if input_method == "Upload FASTA Files":
        protein_a_file = st.sidebar.file_uploader("Upload Protein A FASTA", type=["fasta", "fa", "txt"])
        protein_b_file = st.sidebar.file_uploader("Upload Protein B FASTA", type=["fasta", "fa", "txt"])
        
        if protein_a_file is not None:
            protein_a_content = protein_a_file.getvalue().decode("utf-8")
            protein_a_dict = read_fasta(protein_a_content)
            if protein_a_dict:
                protein_a_sequence = list(protein_a_dict.values())[0]
                st.sidebar.success("Protein A loaded successfully!")
            else:
                st.sidebar.error("Invalid FASTA format for Protein A")
        
        if protein_b_file is not None:
            protein_b_content = protein_b_file.getvalue().decode("utf-8")
            protein_b_dict = read_fasta(protein_b_content)
            if protein_b_dict:
                protein_b_sequence = list(protein_b_dict.values())[0]
                st.sidebar.success("Protein B loaded successfully!")
            else:
                st.sidebar.error("Invalid FASTA format for Protein B")
    
    else:  # Enter FASTA Text
        protein_a_text = st.sidebar.text_area("Enter Protein A FASTA", height=150)
        protein_b_text = st.sidebar.text_area("Enter Protein B FASTA", height=150)
        
        if protein_a_text:
            if protein_a_text.startswith(">"):
                protein_a_dict = read_fasta(protein_a_text)
                if protein_a_dict:
                    protein_a_sequence = list(protein_a_dict.values())[0]
                else:
                    st.sidebar.error("Invalid FASTA format for Protein A")
            else:
                # Assume raw sequence without header
                protein_a_sequence = re.sub(r'[^A-Za-z]', '', protein_a_text)
        
        if protein_b_text:
            if protein_b_text.startswith(">"):
                protein_b_dict = read_fasta(protein_b_text)
                if protein_b_dict:
                    protein_b_sequence = list(protein_b_dict.values())[0]
                else:
                    st.sidebar.error("Invalid FASTA format for Protein B")
            else:
                # Assume raw sequence without header
                protein_b_sequence = re.sub(r'[^A-Za-z]', '', protein_b_text)
    
    # Predict button
    if st.sidebar.button("Predict Similarity Class"):
        if protein_a_sequence and protein_b_sequence:
            with st.spinner("Extracting protein features..."):
                # Extract features
                p1_features = extract_features_from_sequence(protein_a_sequence)
                p2_features = extract_features_from_sequence(protein_b_sequence)
                
                if p1_features and p2_features:
                    # Create feature dataframe
                    df = create_feature_dataframe(p1_features, p2_features)
                    
                    # Expected features in correct order
                    expected_features = [
                        'P1', 'P1_AlipIndex', 'P1_Autocorr', 'P1_Autocov', 'P1_boman',
                        'P1_charge', 'P1_crosscov', 'P1_hydrmom', 'P1_isoelctricp',
                        'P1_instablility', 'P1_massshift', 'P1_molweight', 'P1_mz', 'P1_sc',
                        'P2', 'P2_AlipIndex', 'P2_Autocorr', 'P2_Autocov', 'P2_boman',
                        'P2_charge', 'P2_crosscov', 'P2_hydrmom', 'P2_isoelctricp',
                        'P2_instablility', 'P2_massshift', 'P2_molweight', 'P2_mz', 'P2_sc'
                    ]
                    
                    # Ensure all expected features are present
                    for feature in expected_features:
                        if feature not in df.columns:
                            df[feature] = 0
                    
                    # Reorder columns to match training data
                    df = df[expected_features]
                    
                    try:
                        # Load the model
                        with st.spinner("Making prediction..."):
                            model = joblib.load('models/random_forest_model.joblib')
                            prediction = model.predict(df)[0]
                            
                            # Get class and description
                            class_value, class_description = get_similarity_class(prediction)
                            
                            # Display results
                            st.success(f"Prediction Completed!")
                            
                            # Create results display
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Similarity Score", f"{prediction:.1f}%")
                            with col2:
                                st.metric("Similarity Class", f"Class {class_value}")
                            
                            st.info(f"Classification: {class_description}")
                            
                            # Show progress bar
                            st.progress(prediction/100)
                            
                            # Display sequences
                            st.subheader("Protein Sequences")
                            p1_col, p2_col = st.columns(2)
                            
                            with p1_col:
                                st.write("**Protein A:**")
                                display_seq = protein_a_sequence[:100] + "..." if len(protein_a_sequence) > 100 else protein_a_sequence
                                st.code(display_seq)
                                st.caption(f"Length: {len(protein_a_sequence)} amino acids")
                            
                            with p2_col:
                                st.write("**Protein B:**")
                                display_seq = protein_b_sequence[:100] + "..." if len(protein_b_sequence) > 100 else protein_b_sequence
                                st.code(display_seq)
                                st.caption(f"Length: {len(protein_b_sequence)} amino acids")
                            
                            # Show some key features
                            with st.expander("View Key Protein Features"):
                                feature_df = pd.DataFrame({
                                    'Feature': ['Length', 'Molecular Weight', 'Isoelectric Point', 'Instability Index', 'Aliphatic Index'],
                                    'Protein A': [p1_features['length'], f"{p1_features['molweight']:.1f}", f"{p1_features['isoelctricp']:.2f}", 
                                                f"{p1_features['instablility']:.2f}", f"{p1_features['AlipIndex']:.1f}"],
                                    'Protein B': [p2_features['length'], f"{p2_features['molweight']:.1f}", f"{p2_features['isoelctricp']:.2f}", 
                                                f"{p2_features['instablility']:.2f}", f"{p2_features['AlipIndex']:.1f}"]
                                })
                                st.table(feature_df)
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        import traceback
                        st.error(f"Detailed error: {traceback.format_exc()}")
                else:
                    st.error("Could not extract features from one or both protein sequences.")
        else:
            st.warning("Please provide both protein sequences to make a prediction.")
    
    # Information section when no prediction is made
    if "prediction" not in locals():
        st.info("👈 Upload or enter protein sequences and click 'Predict Similarity Class'")
        
        st.markdown("""
        ### How this works:
        
        1. Input two protein sequences in FASTA format
        2. The app extracts biochemical and physical features from both proteins
        3. These features are fed into a trained Random Forest model
        4. The model predicts a similarity score between 1-100
        5. This score is mapped to a similarity class (0-9)
        
        ### Similarity Classes:
        
        - **Class 0**: Very Low Similarity (1-10%)
        - **Class 1**: Low Similarity (11-20%)
        - **Class 2**: Low-Moderate Similarity (21-30%)
        - **Class 3**: Moderate Similarity (31-40%)
        - **Class 4**: Moderate Similarity (41-50%)
        - **Class 5**: Moderate-High Similarity (51-60%)
        - **Class 6**: High Similarity (61-70%)
        - **Class 7**: High Similarity (71-80%)
        - **Class 8**: Very High Similarity (81-90%)
        - **Class 9**: Extremely High Similarity (91-100%)
        """)

if __name__ == "__main__":
    main()
