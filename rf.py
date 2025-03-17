import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Set page configuration
st.set_page_config(
    page_title="Protein Similarity Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS - forcing white background and removing extra elements
st.markdown("""
<style>
    /* Force white background throughout the app */
    .stApp {
        background-color: white;
    }
    
    /* Remove default dark mode detection */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: white;
            color: black;
        }
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        color: #1A5276;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.4rem;
        color: #2E86C1;
        margin-top: 1.5rem;
    }
    
    /* Results container */
    .results-container {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #AED6F1;
        margin-top: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2874A6;
        color: white;
    }
    
    /* Result class display */
    .result-class {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background-color: white;
        border-radius: 8px;
        margin: 20px 0;
        border: 1px solid #D5D8DC;
    }
    
    /* Override any potential dark mode elements */
    .st-bq, .st-bb, .st-bc, .st-bd, .st-be, .st-bf, .st-bg, .st-bh {
        color: black;
    }
    
    .st-g0, .st-g1, .st-g2, .st-g3, .st-g4, .st-g5, .st-g6, .st-g7, .st-g8, .st-g9 {
        background-color: white;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Function to map similarity scores to classes with descriptions
def map_values(value):
    classes = {
        (1, 10): {"class": 0, "description": "Very Low Similarity"},
        (11, 20): {"class": 1, "description": "Low Similarity"},
        (21, 30): {"class": 2, "description": "Low-Moderate Similarity"},
        (31, 40): {"class": 3, "description": "Moderate Similarity"},
        (41, 50): {"class": 4, "description": "Moderate-High Similarity"},
        (51, 60): {"class": 5, "description": "High Similarity"},
        (61, 70): {"class": 6, "description": "High-Very High Similarity"},
        (71, 80): {"class": 7, "description": "Very High Similarity"},
        (81, 90): {"class": 8, "description": "Extremely High Similarity"},
        (91, 100): {"class": 9, "description": "Near-Perfect Match"}
    }
    
    for (min_val, max_val), info in classes.items():
        if min_val <= value <= max_val:
            return info
    
    return {"class": -1, "description": "Unknown"}

# Function to read FASTA format
def read_fasta(file_content):
    fasta_dict = {}
    current_key = None
    current_sequence = []

    for line in file_content:
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
        return {"class": 9, "description": "Perfect Match", "score": 100, "perfect_match": True}

    try:
        # Load the Random Forest model
        model = joblib.load('models/random_forest_model (2).joblib')

        # Prepare data for model input
        sequences = [protein_a_sequence, protein_b_sequence]

        # Vectorize the sequences
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=26)
        X = vectorizer.fit_transform(sequences)

        # Predict similarity using the model
        similarity_score = model.predict(X)[0]  

        # Scale the score from 0-10 to 0-100
        scaled_similarity_score = similarity_score * 10  

        # Map to similarity class
        result = map_values(scaled_similarity_score)
        result["score"] = scaled_similarity_score
        result["perfect_match"] = False
        
        return result
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return {"class": -1, "description": "Error", "score": 0, "perfect_match": False}

# Main function
def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Protein Similarity Analyzer</h1>', unsafe_allow_html=True)
    
    # Brief description
    st.markdown("""
    This tool analyzes the similarity between two protein sequences using a machine learning model.
    Upload your sequences in FASTA format or paste them directly in the input fields.
    """)
    
    # Create two columns layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Input Methods")
        
        # Create tabs for different input methods
        input_tab1, input_tab2 = st.tabs(["File Upload", "Text Input"])
        
        with input_tab1:
            st.markdown("#### Upload FASTA Files")
            protein_a_file = st.file_uploader("Protein A File", type=["fasta", "txt"])
            protein_b_file = st.file_uploader("Protein B File", type=["fasta", "txt"])
        
        with input_tab2:
            st.markdown("#### Enter Sequences")
            protein_a_input = st.text_area("Protein A Sequence (FASTA format)", height=120)
            protein_b_input = st.text_area("Protein B Sequence (FASTA format)", height=120)
        
        # Submit button with spinner
        analyze_button = st.button("Analyze Similarity")
    
    with col2:
        # Main content area for results
        if analyze_button:
            with st.spinner("Analyzing protein sequences..."):
                # Extract sequences from inputs
                protein_a_sequence = None
                protein_b_sequence = None
                
                # Process Protein A
                if protein_a_file:
                    protein_a_content = protein_a_file.getvalue().decode("utf-8")
                    protein_a_dict = read_fasta(protein_a_content.splitlines())
                    if protein_a_dict:
                        protein_a_sequence = list(protein_a_dict.values())[0]
                        protein_a_name = list(protein_a_dict.keys())[0]
                    else:
                        st.error("Could not parse Protein A file as FASTA format.")
                elif protein_a_input:
                    try:
                        protein_a_dict = read_fasta(protein_a_input.splitlines())
                        if protein_a_dict:
                            protein_a_sequence = list(protein_a_dict.values())[0]
                            protein_a_name = list(protein_a_dict.keys())[0]
                        else:
                            # Assume direct sequence input without FASTA header
                            protein_a_sequence = protein_a_input.strip()
                            protein_a_name = "Protein A"
                    except:
                        protein_a_sequence = protein_a_input.strip()
                        protein_a_name = "Protein A"
                
                # Process Protein B
                if protein_b_file:
                    protein_b_content = protein_b_file.getvalue().decode("utf-8")
                    protein_b_dict = read_fasta(protein_b_content.splitlines())
                    if protein_b_dict:
                        protein_b_sequence = list(protein_b_dict.values())[0]
                        protein_b_name = list(protein_b_dict.keys())[0]
                    else:
                        st.error("Could not parse Protein B file as FASTA format.")
                elif protein_b_input:
                    try:
                        protein_b_dict = read_fasta(protein_b_input.splitlines())
                        if protein_b_dict:
                            protein_b_sequence = list(protein_b_dict.values())[0]
                            protein_b_name = list(protein_b_dict.keys())[0]
                        else:
                            # Assume direct sequence input without FASTA header
                            protein_b_sequence = protein_b_input.strip()
                            protein_b_name = "Protein B"
                    except:
                        protein_b_sequence = protein_b_input.strip()
                        protein_b_name = "Protein B"
                
                # Validate input
                if not protein_a_sequence or not protein_b_sequence:
                    st.error("Please provide valid protein sequences for both inputs.")
                else:
                    # Calculate similarity
                    similarity_result = calculate_similarity_class(protein_a_sequence, protein_b_sequence)
                    
                    # Display results - NO BAR, just the header text
                    st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)
                    
                    # Display only the class (and score only if perfect match)
                    if similarity_result["perfect_match"]:
                        st.markdown(f'<div class="result-class">100% Similar</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-class">Class: {similarity_result["class"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'**Classification**: {similarity_result["description"]}')
                    
                    # Show sequence comparison
                    st.markdown("### Sequence Information")
                    seq_col1, seq_col2 = st.columns(2)
                    
                    with seq_col1:
                        st.markdown(f"**{protein_a_name}**")
                        st.text_area("Protein A", protein_a_sequence, height=100, disabled=True)
                        st.text(f"Length: {len(protein_a_sequence)} amino acids")
                        
                    with seq_col2:
                        st.markdown(f"**{protein_b_name}**")
                        st.text_area("Protein B", protein_b_sequence, height=100, disabled=True)
                        st.text(f"Length: {len(protein_b_sequence)} amino acids")
                    
                    # Simple information about classes
                    with st.expander("About Similarity Classes"):
                        st.markdown("""
                        - **Class 0-2**: Low similarity (1-30%)
                        - **Class 3-4**: Moderate similarity (31-50%)
                        - **Class 5-6**: High similarity (51-70%)
                        - **Class 7-9**: Very high similarity (71-100%)
                        """)
        else:
            # Show instructions when no analysis has been performed
            st.markdown("## How to Use")
            st.markdown("""
            1. Upload FASTA files or paste protein sequences in the input fields
            2. Click 'Analyze Similarity' to calculate the similarity score
            3. View the similarity class result
            """)
            
            st.markdown("## About This Tool")
            st.markdown("""
            This tool uses a Random Forest machine learning model to predict the similarity 
            between two protein sequences. The model analyzes n-gram patterns in amino acid 
            sequences to determine structural and functional similarity.
            """)
            
            # Simple placeholder
            st.markdown("### Results will appear here after analysis")
            st.image("https://via.placeholder.com/800x200?text=Upload+Protein+Sequences+to+See+Results", use_container_width=True)

# Run the application
if __name__ == "__main__":
    main()
