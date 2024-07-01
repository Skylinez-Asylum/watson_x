import streamlit as st
import fitz  # PyMuPDF
from bias_finder import BiasFinder
# Function to send input to backend for bias detection
import nltk
from summarizer import Summary
from bias_scoring import BiasScoring
from bias_explainer import BiasExplainer
#nltk.download('punkt') 
def analyze_bias(input_text, option):
    payload = {
        "text": input_text,
        "option": option
    }
    bias_finder=BiasFinder()
    response =bias_finder.run(input_text)
    return response

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def summarizer(text):
    summarize = Summary()
    return summarize.run(text)

def ranking(text,bias):
    ranker = BiasScoring()
    return ranker.run(text,bias)
BE=BiasExplainer()


# Streamlit UI
def main():
    st.title("Bias Detection and Correction App")

    
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file is not None:
        input_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Document Content", input_text, height=300)
        option="fileupload"
        if st.button("Analyze Document"):
            try:
                summary_sum = summarizer(input_text)
                result = analyze_bias(input_text, option)
                st.write("Detected Biases:")
                for bias, sentences in result.items():
                    st.header(f"**{bias}:**")
                    for sentence in sentences:
                        st.subheader("Sentence :")
                        st.write(f"- {sentence}\t\t\t.\n The rank is {ranking(sentence,bias)}")
                        be=BE.run(sentence,bias,summary_sum)
                        st.subheader("Bias explained for the above sentence")
                        st.write(be)
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
