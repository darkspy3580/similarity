import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Protein Similarity Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
    }
    .results-container {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .sidebar-content {
        padding: 20px 10px;
    }
    .stButton>button {
        background-color: #1976D2;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .similarity-class {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
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
        return {"class": 9, "description": "Perfect Match", "score": 100}

    try:
        # Load the Random Forest model
        model = joblib.load('models/random_forest_model.joblib')

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
        
        return result
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return {"class": -1, "description": "Error", "score": 0}

# Function to generate visualization of the similarity score
def generate_similarity_gauge(similarity_score):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create gauge chart
    gauge_range = np.linspace(0, 100, 100)
    gauge_height = np.ones(100)
    
    # Create color gradient
    colors = sns.color_palette("RdYlGn", 100)
    
    # Plot the gauge
    plt.barh(0, gauge_range, height=0.6, color=colors)
    
    # Add marker for the current score
    plt.scatter(similarity_score, 0, s=300, color='darkblue', zorder=5, marker='v')
    
    # Customize the plot
    plt.xlim(0, 100)
    plt.ylim(-0.5, 0.5)
    plt.axis('off')
    
    # Add score text
    plt.text(similarity_score, -0.1, f"{similarity_score:.1f}%", 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add labels
    plt.text(5, 0.2, "Low", ha='center', fontsize=10)
    plt.text(50, 0.2, "Medium", ha='center', fontsize=10)
    plt.text(95, 0.2, "High", ha='center', fontsize=10)
    
    return fig

# Function to format sequence for display
def format_sequence_display(sequence, max_length=50):
    if len(sequence) <= max_length:
        return sequence
    return sequence[:max_length] + "..."

# Helper function to create info boxes
def show_info_box(title, content):
    st.markdown(f"""
    <div class="info-box">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### Input Methods")
        
        # Create tabs for different input methods
        input_tab1, input_tab2 = st.tabs(["File Upload", "Text Input"])
        
        with input_tab1:
            st.markdown("#### Upload FASTA Files")
            protein_a_file = st.file_uploader("Protein A File", type=["fasta", "txt"])
            protein_b_file = st.file_uploader("Protein B File", type=["fasta", "txt"])
        
        with input_tab2:
            st.markdown("#### Enter Sequences")
            protein_a_input = st.text_area("Protein A Sequence (FASTA format)", height=150)
            protein_b_input = st.text_area("Protein B Sequence (FASTA format)", height=150)
        
        # Submit button with spinner
        analyze_button = st.button("Analyze Similarity", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
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
                    
                    # Display results
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)
                    
                    # Create three columns for results display
                    res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
                    
                    with res_col1:
                        st.metric("Similarity Score", f"{similarity_result['score']:.1f}%")
                        
                    with res_col2:
                        st.metric("Similarity Class", similarity_result['class'])
                        
                    with res_col3:
                        st.metric("Classification", similarity_result['description'])
                    
                    # Display similarity gauge
                    st.markdown("### Similarity Visualization")
                    gauge_fig = generate_similarity_gauge(similarity_result['score'])
                    st.pyplot(gauge_fig)
                    
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
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Additional information
                    with st.expander("About Similarity Classes"):
                        similarity_classes = {
                            0: "Very Low Similarity (1-10%): Proteins are likely unrelated.",
                            1: "Low Similarity (11-20%): Minimal structural or functional similarity.",
                            2: "Low-Moderate Similarity (21-30%): Some similarities exist but likely different functions.",
                            3: "Moderate Similarity (31-40%): May share some functional domains.",
                            4: "Moderate-High Similarity (41-50%): Possibly related proteins with some conserved regions.",
                            5: "High Similarity (51-60%): Likely related proteins with similar functions.",
                            6: "High-Very High Similarity (61-70%): Strongly related proteins, probably homologous.",
                            7: "Very High Similarity (71-80%): Closely related proteins with conserved functions.",
                            8: "Extremely High Similarity (81-90%): Very closely related, likely isoforms or variants.",
                            9: "Near-Perfect Match (91-100%): Almost identical proteins."
                        }
                        
                        for class_num, desc in similarity_classes.items():
                            st.markdown(f"**Class {class_num}**: {desc}")
        else:
            # Show instructions when no analysis has been performed
            show_info_box("How to Use", """
            1. Upload FASTA files or paste protein sequences in the input fields
            2. Click 'Analyze Similarity' to calculate the similarity score
            3. View detailed results and visualizations
            """)
            
            show_info_box("About This Tool", """
            This tool uses a Random Forest machine learning model to predict the similarity 
            between two protein sequences. The model analyzes n-gram patterns in amino acid 
            sequences to determine structural and functional similarity.
            """)
            
            # Placeholder for results area
            st.image("https://via.placeholder.com/800x400?text=Upload+Protein+Sequences+to+See+Results", use_column_width=True)

# Run the application
if __name__ == "__main__":
    main()
