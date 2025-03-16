import streamlit as st
import numpy as np
import joblib
import re
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight
from Bio import SeqIO
from io import StringIO

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

def extract_protein_features(seq1, seq2):
    """Extract the specific features used in training the RF model"""
    features = {}
    
    # Clean sequences to ensure valid amino acids
    valid_aa = "ACDEFGHIKLMNPQRSTVWY"
    seq1 = ''.join(aa for aa in seq1.upper() if aa in valid_aa)
    seq2 = ''.join(aa for aa in seq2.upper() if aa in valid_aa)
    
    # Initialize Bio.SeqUtils.ProtParam objects for analysis
    try:
        prot1 = ProteinAnalysis(seq1)
        prot2 = ProteinAnalysis(seq2)
        
        # P1 features
        features['P1'] = len(seq1)
        features['P1_AlipIndex'] = prot1.aliphatic_index()
        features['P1_Autocorr'] = np.mean(prot1.protein_scale(window=7, param_dict='Flex'))
        features['P1_Autocov'] = np.var(prot1.protein_scale(window=7, param_dict='Flex'))
        features['P1_boman'] = sum(prot1.protein_scale(window=7, param_dict='hw'))
        features['P1_charge'] = prot1.charge_at_pH(7.0)
        features['P1_crosscov'] = 0  # Placeholder, compute if needed
        features['P1_hydrmom'] = prot1.gravy()
        features['P1_isoelctricp'] = prot1.isoelectric_point()
        features['P1_instablility'] = prot1.instability_index()
        features['P1_massshift'] = sum(prot1.protein_scale(window=7, param_dict='Flex'))
        features['P1_molweight'] = prot1.molecular_weight()
        features['P1_mz'] = prot1.molecular_weight() / abs(prot1.charge_at_pH(7.0)) if prot1.charge_at_pH(7.0) != 0 else 0
        features['P1_sc'] = len([aa for aa in seq1 if aa in 'DEHKR']) / max(1, len(seq1))
        
        # P2 features
        features['P2'] = len(seq2)
        features['P2_AlipIndex'] = prot2.aliphatic_index()
        features['P2_Autocorr'] = np.mean(prot2.protein_scale(window=7, param_dict='Flex'))
        features['P2_Autocov'] = np.var(prot2.protein_scale(window=7, param_dict='Flex'))
        features['P2_boman'] = sum(prot2.protein_scale(window=7, param_dict='hw'))
        features['P2_charge'] = prot2.charge_at_pH(7.0)
        features['P2_crosscov'] = 0  # Placeholder, compute if needed
        features['P2_hydrmom'] = prot2.gravy()
        features['P2_isoelctricp'] = prot2.isoelectric_point()
        features['P2_instablility'] = prot2.instability_index()
        features['P2_massshift'] = sum(prot2.protein_scale(window=7, param_dict='Flex'))
        features['P2_molweight'] = prot2.molecular_weight()
        features['P2_mz'] = prot2.molecular_weight() / abs(prot2.charge_at_pH(7.0)) if prot2.charge_at_pH(7.0) != 0 else 0
        features['P2_sc'] = len([aa for aa in seq2 if aa in 'DEHKR']) / max(1, len(seq2))
        
    except Exception as e:
        st.error(f"Error calculating protein features: {str(e)}")
        return None
    
    return features

def get_rf_class(prediction_value):
    """Convert the RF model prediction to the correct class"""
    if 1 <= prediction_value <= 10:
        return 0
    elif 11 <= prediction_value <= 20:
        return 1
    elif 21 <= prediction_value <= 30:
        return 2
    elif 31 <= prediction_value <= 40:
        return 3
    elif 41 <= prediction_value <= 50:
        return 4
    elif 51 <= prediction_value <= 60:
        return 5
    elif 61 <= prediction_value <= 70:
        return 6
    elif 71 <= prediction_value <= 80:
        return 7
    elif 81 <= prediction_value <= 90:
        return 8
    elif 91 <= prediction_value <= 100:
        return 9
    else:
        return None

def get_class_description(class_value):
    """Return human-readable description for each class"""
    descriptions = {
        0: "Very Low Similarity (1-10%)",
        1: "Low Similarity (11-20%)",
        2: "Low-Moderate Similarity (21-30%)",
        3: "Moderate Similarity (31-40%)",
        4: "Moderate Similarity (41-50%)",
        5: "Moderate-High Similarity (51-60%)",
        6: "High Similarity (61-70%)",
        7: "High Similarity (71-80%)",
        8: "Very High Similarity (81-90%)",
        9: "Extremely High Similarity (91-100%)"
    }
    return descriptions.get(class_value, "Unknown Classification")

def main():
    st.title("Protein Similarity Using Random Forest Model")
    st.write("Upload or enter two protein sequences to predict their similarity class.")
    
    # Sidebar for uploading protein sequences
    st.sidebar.title("Upload Protein Sequences")
    protein_a_file = st.sidebar.file_uploader("Upload FASTA file for Protein A", type=["fasta", "fa", "txt"])
    protein_b_file = st.sidebar.file_uploader("Upload FASTA file for Protein B", type=["fasta", "fa", "txt"])
    
    # Text input alternatives
    st.sidebar.markdown("### Or enter sequences directly:")
    protein_a_input = st.sidebar.text_area("Enter Protein A sequence (FASTA format)")
    protein_b_input = st.sidebar.text_area("Enter Protein B sequence (FASTA format)")
    
    # Submit button
    if st.sidebar.button("Predict Similarity"):
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
        
        # Validate sequences and predict similarity using RF model
        if protein_a_sequence and protein_b_sequence:
            try:
                # Extract features for RF model
                features = extract_protein_features(protein_a_sequence, protein_b_sequence)
                
                if features:
                    # Load the trained model
                    try:
                        model = joblib.load('models/random_forest_model.joblib')
                        
                        # Create DataFrame with the features in the correct order
                        feature_cols = ['P1', 'P1_AlipIndex', 'P1_Autocorr', 'P1_Autocov', 'P1_boman',
                                        'P1_charge', 'P1_crosscov', 'P1_hydrmom', 'P1_isoelctricp',
                                        'P1_instablility', 'P1_massshift', 'P1_molweight', 'P1_mz', 'P1_sc',
                                        'P2', 'P2_AlipIndex', 'P2_Autocorr', 'P2_Autocov', 'P2_boman',
                                        'P2_charge', 'P2_crosscov', 'P2_hydrmom', 'P2_isoelctricp',
                                        'P2_instablility', 'P2_massshift', 'P2_molweight', 'P2_mz', 'P2_sc']
                        
                        # Create DataFrame with just the features needed (excluding similaritry score)
                        df = pd.DataFrame([features])
                        
                        # Make sure the DataFrame has all needed columns in the right order
                        for col in feature_cols:
                            if col not in df.columns:
                                df[col] = 0  # Default value for missing columns
                        
                        # Select only the columns used during training and in the right order
                        df = df[feature_cols]
                        
                        # Make prediction
                        prediction_value = model.predict(df)[0]
                        
                        # Get class based on prediction value
                        class_value = get_rf_class(prediction_value)
                        class_description = get_class_description(class_value)
                        
                        # Display results
                        st.success(f"Prediction Value: {prediction_value:.2f}")
                        st.info(f"Similarity Class: {class_value} - {class_description}")
                        
                        # Create a visual indicator of similarity (progress bar)
                        similarity_percentage = prediction_value
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
                        st.error(f"Model prediction error: {str(e)}")
                        import traceback
                        st.error(f"Detailed error: {traceback.format_exc()}")
                else:
                    st.error("Could not extract protein features for prediction.")
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
        else:
            st.warning("Please provide both protein sequences to predict similarity.")
    
    # Add some informational content to the main page when no calculation is performed
    if "prediction_value" not in locals():
        st.info("⬅️ Please upload or enter protein sequences in the sidebar and click 'Predict Similarity'")
        st.markdown("""
        ### How it works:
        1. Upload FASTA files or enter protein sequences in FASTA format
        2. Click 'Predict Similarity' to analyze using the Random Forest model
        3. View the similarity class (0-9) and description
        
        **Features used by the model include:**
        - Protein length
        - Aliphatic index
        - Autocorrelation
        - Autocovariance
        - Boman index
        - Charge
        - Cross-covariance
        - Hydrophobic moment
        - Isoelectric point
        - Instability index
        - Mass shift
        - Molecular weight
        - Mass-to-charge ratio
        - Side chain properties
        """)

if __name__ == "__main__":
    main()
