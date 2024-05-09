
import gradio as gr  # Importing Gradio library for building UI
from DocumentVector import DocumentLoader  # Importing DocumentLoader class for loading PDF documents
from ResponseRetrieval import ResponseRetrieval  # Importing ResponseRetrieval class for response retrieval

def simpleUI(pdf_path, question, threshold):
    retrieval_obj = ResponseRetrieval(pdf_path)  # Initialize ResponseRetrieval object
    answer = retrieval_obj.answer(question, threshold)  # Get response to the question
    return answer

# Create Gradio interface for the response retrieval
demo = gr.Interface(
    fn=simpleUI,
    inputs=["text", "text", gr.Radio(["Precise", "Balanced", "Creative"], label="Style")],
    outputs=["text"],
)

# Launch the Gradio interface
demo.launch()