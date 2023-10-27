import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
import torch

from html_templates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # creates a pdf object with pages
        pdf_reader = PdfReader(pdf)
        print(f"reading pdf: {pdf_reader.metadata.get('/Title')}")
        for page in tqdm(pdf_reader.pages, desc="pages"):
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, use_open_ai):
    if use_open_ai:
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    else:
        # this is doing this on the cpu
        embedding = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl",
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using {device=}")
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embedding,
    )
    return vectorstore


def get_conversation_chain(vectorstore, use_open_ai):
    if use_open_ai:
        llm = ChatOpenAI()
    else:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 512},
        )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation(
        {
            "question": user_question,
        }
    )
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace(
                    "{{MSG}}",
                    message.content,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace(
                    "{{MSG}}",
                    message.content,
                ),
                unsafe_allow_html=True,
            )

    # to see the format of this response
    # st.write(response)


# The general way this thing works
# the streamlit app runs persistently
# when the user does something, it triggers one of those inputs
# The messages are actually covered by some custom HTML templates
# that are displayed as the messages come in
# The conversation is managed by ConversationBufferMemory and
# ConversationalRetrievalChain.


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon="📚",
    )

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs 📖")
    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        # whatever you add here goes in the sidebar
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here as click Process",
            accept_multiple_files=True,
        )
        # add a little toggle to turn on or off openAI, since
        # it costs money
        # if off, this downloads and runs HuggingFace models
        # that can take a while if it is on CPU
        use_open_ai = st.toggle("Use OpenAI (💰💰💰)")
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create the vector store with the embeddings
                vectorstore = get_vectorstore(
                    text_chunks,
                    use_open_ai=use_open_ai,
                )
                # create the conversation chain, keep it persistent
                st.session_state.conversation = get_conversation_chain(
                    vectorstore,
                    use_open_ai=use_open_ai,
                )


if __name__ == "__main__":
    main()
