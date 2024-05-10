import os
import shutil
from dotenv import load_dotenv
import openai
from DocumentVector import DocumentLoader, VectorDBBuilder # Importing DocumentLoader class for loading PDF documents
from langchain.chat_models import ChatOpenAI  # Importing ChatOpenAI for chat-based models
from langchain.memory import ConversationBufferMemory  # Importing ConversationBufferMemory for conversation memory
from langchain.chains import ConversationalRetrievalChain  # Importing ConversationalRetrievalChain for conversational retrieval

class ResponseRetrieval:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def answer(self, question, threshold):
        # Load OpenAI API key from environment variables
        load_dotenv()
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        # Determine temperature based on threshold for OpenAI response generation
        temperature = self._get_temperature(threshold)
        # Initialize OpenAI chat model
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
        # Initialize conversation memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Load PDF document
        DocumentLoader_obj = DocumentLoader(self.pdf_path)
        # Create retriever from the document vector database
        VectorDBBuilder_obj=VectorDBBuilder(DocumentLoader_obj.load_pages())
        retriever = VectorDBBuilder_obj.build().as_retriever(search_type="similarity")
        # Initialize ConversationalRetrievalChain with OpenAI model, retriever, and memory
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
        # Generate response to the question
        result = qa({"question": question})
        return result.get('answer', None)

    def _get_temperature(self, threshold):
        # Assign temperature based on user's desired threshold
        if threshold == 'Precise':
            return 0
        elif threshold == 'Balanced':
            return 0.5
        else:
            return 1

# Example usage:
# user_input = ''
# n = 0
# retrieval_obj = ResponseRetrieval('sample_pdf.pdf')
# while user_input != 'endConv':
#     if n == 0:
#         user_input = input('Please enter the question: ')
#     print(retrieval_obj.answer(user_input, 'Balanced'))
#     user_input = input('Enter "finish" to end the conversation, otherwise continue: ')
#     if user_input.lower() == 'finish':
#         user_input = 'endConv'
#     else:
#         n += 1


