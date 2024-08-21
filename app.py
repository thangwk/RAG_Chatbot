import os
import tempfile
import streamlit as st
from streamlit_chat import message
#from rag import ChatCSV
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

class ChatCSV:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        """
        Initializes the question-answering system with default configurations.
        This constructor sets up the following components:
        - A ChatOllama model for generating responses ('neural-chat').
        - A RecursiveCharacterTextSplitter for splitting text into chunks.
        - A PromptTemplate for constructing prompts with placeholders for question and context.
        """
        # Initialize the ChatOllama model with 'neural-chat'.
        self.model = ChatOllama(base_url = 'http://54.255.10.70:11434', model= 'llama2')

        # Initialize the RecursiveCharacterTextSplitter with specific chunk settings.
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        # Initialize the PromptTemplate with a predefined template for constructing prompts.
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are a helpful nutritionist that analyses different ingredients to come up with a meal plan.
            Use the following pieces of retrieved context to answer the question.
            Give Name when possible. If you don't know the answer,
            just say that you don't know.  [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        
    def ingest(self, csv_file_path: str):
        '''
        Ingests data from a CSV file containing resumes, process the data, and set up the
        components for further analysis.
        Parameters:
        - csv_file_path (str): The file path to the CSV file.
        Usage:
        obj.ingest("/path/to/data.csv")
        This function uses a CSVLoader to load the data from the specified CSV file.
        Args:
        - file.path (str): The path to the CSV file.
        - encoding (str): The character encoding of the file (default is 'utf-8').
        - source_column (str): The column in the CSV containing the data (default is "Resume").
        '''        
        loader = CSVLoader(
            file_path=csv_file_path,
            # file_path = "D:\Python Projects\RAG_LLM\Menu.csv"
            encoding='utf-8',
            # source_column="Resume"
            csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['NAME','COST','Kcal','Fat','Carb','Protein','Fiber','MealSize','Ingrediants Needed']
            }
            )
        
        # loads the data
        data = loader.load()

        # splits the documents into chunks
        chunks = self.text_splitter.split_documents(data)
        chunks = filter_complex_metadata(chunks)

        # creates a vector store using embedding
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        # sets up the retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        # Define a processing chain for handling a question-answer scenario.
        # The chain consists of the following components:
        # 1. "context" from the retriever
        # 2. A passthrough for the "question"
        # 3. Processing with the "prompt"
        # 4. Interaction with the "model"
        # 5. Parsing the output using the "StrOutputParser"
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
    def ask(self, query: str):
        """
        Asks a question using the configured processing chain.
        Parameters:
        - query (str): The question to be asked.
        Returns:
        - str: The result of processing the question through the configured chain.
        If the processing chain is not set up (empty), a message is returned
        prompting to add a CSV document first.
        """
        if not self.chain:
            return "Please, add a CSV document first."

        return self.chain.invoke(query)

    def clear(self):
        """
        Clears the components in the question-answering system.
        This method resets the vector store, retriever, and processing chain to None,
        effectively clearing the existing configuration.
        """
        # Set the vector store to None.
        self.vector_store = None

        # Set the retriever to None.
        self.retriever = None

        # Set the processing chain to None.
        self.chain = None
# adds a title for the web page
st.set_page_config(page_title="Resume Chatbot")

def display_messages():
    """
    Displays chat messages in the Streamlit app.
    This function assumes that chat messages are stored in the Streamlit session state
    under the key "messages" as a list of tuples, where each tuple contains the message
    content and a boolean indicating whether it's a user message or not.
    Additionally, it creates an empty container for a thinking spinner in the Streamlit
    session state under the key "thinking_spinner".
    Note: Streamlit (st) functions are used for displaying content in a Streamlit app.
    """
    # Display a subheader for the chat.
    st.subheader("Chat")

    # Iterate through messages stored in the session state.
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        # Display each message using the message function with appropriate styling.
        message(msg, is_user=is_user, key=str(i))

    # Create an empty container for a thinking spinner and store it in the session state.
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    """
    Processes user input and updates the chat messages in the Streamlit app.
    This function assumes that user input is stored in the Streamlit session state
    under the key "user_input," and the question-answering assistant is stored
    under the key "assistant."
    Additionally, it utilizes Streamlit functions for displaying a thinking spinner
    and updating the chat messages.
    Note: Streamlit (st) functions are used for interacting with the Streamlit app.
    """
    # Check if there is user input and it is not empty.
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        # Extract and clean the user input.
        user_text = st.session_state["user_input"].strip()

        # Display a thinking spinner while the assistant processes the input.
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            # Ask the assistant for a response based on the user input.
            agent_text = st.session_state["assistant"].ask(user_text)

        # Append user and assistant messages to the chat messages in the session state.
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    """
    Reads and saves the uploaded file, performs ingestion, and clears the assistant state.
    This function assumes that the question-answering assistant is stored in the Streamlit
    session state under the key "assistant," and file-related information is stored under
    the key "file_uploader."
    Additionally, it utilizes Streamlit functions for displaying spinners and updating the
    assistant's state.
    Note: Streamlit (st) functions are used for interacting with the Streamlit app.
    """
    # Clear the state of the question-answering assistant.
    st.session_state["assistant"].clear()

    # Clear the chat messages and user input in the session state.
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    # Iterate through the uploaded files in the session state.
    for file in st.session_state["file_uploader"]:
        # Save the file to a temporary location and get the file path.
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # Display a spinner while ingesting the file.
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():
    """
    Defines the content of the Streamlit app page for ChatCSV.
    This function sets up the initial session state if it doesn't exist and displays
    the main components of the Streamlit app, including the header, file uploader,
    and associated functionalities.
    Note: Streamlit (st) functions are used for interacting with the Streamlit app.
    """
    # Check if the session state is empty (first time loading the app).
    if len(st.session_state) == 0:
        # Initialize the session state with empty chat messages and a ChatCSV assistant.
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatCSV()

    # Display the main header of the Streamlit app.
    st.header("ChatCSV")

    # Display a subheader and a file uploader for uploading CSV files.
    st.subheader("Upload a csv file")
    st.file_uploader(
        "Upload csv",
        type=["csv"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    # Create an empty container for a spinner related to file ingestion
    # and store it in the Streamlit session state under the key "ingestion_spinner".
    st.session_state["ingestion_spinner"] = st.empty()

    # Display chat messages in the Streamlit app using the defined function.
    display_messages()

    # Display a text input field for user messages in the Streamlit app.
    # The input field has a key "user_input," and the on_change event triggers the
    # "process_input" function when the input changes.
    st.text_input("Message", key="user_input", on_change=process_input)

# Check if the script is being run as the main module.
if __name__ == "__main__":
    # Call the "page" function to set up and run the Streamlit app.
    page()