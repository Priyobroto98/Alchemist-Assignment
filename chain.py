
import streamlit as st
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from streamlit_chat import message
from typing import List
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Constants
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
model_type = os.getenv('MODEL_TYPE')

# Initialize LLM and Embeddings
llm = ChatGroq(
    model=model_type,
    temperature=0.7,
    api_key=os.getenv('GROQ_API_KEY')
)

def create_chains(llm, retriever):
    """Create necessary chains for the RAG pipeline."""
    contextualize_ques_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Given a chat history and the latest user question
        which might reference context in the chat history,
        formulate a standalone question which can be understood
        without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is.
        """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    contextualize_chain = contextualize_ques_prompt | llm | StrOutputParser()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_ques_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful RAG AI assistant. 
        Use the following context to answer the user's question. 
        Answer to the point,precise and in a very specific manner.
        If it is an out-of-context question, do not answer and say it is out of context.
        """),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def get_last_k_messages(chat_history, k=5):
    """Get the last k messages from the chat history."""
    return chat_history[-k:] if len(chat_history) > k else chat_history


def create_generic_prompt_chain(llm):
    """Create a chain to handle vague/generic inputs."""
    generic_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        The user's input seems vague or generic. Your task is to guide the user
        to provide a more specific input. Do NOT answer the query; instead, 
        politely ask for clarification or specificity.
        """),
        ("human", "{input}")
    ])
    generic_chain = LLMChain(llm=llm, prompt=generic_prompt)
    return generic_chain

def router_logic(llm, user_input):
    """Decides if the input is 'specific' or 'generic'."""
    router_prompt = PromptTemplate(
        input_variables=["input"],
        template="""
        Based on the input provided, classify it as either 'specific' or 'generic':
        Input: {input}
        Respond with either 'specific' or 'generic' only.
        """
    )
    router_chain = LLMChain(llm=llm, prompt=router_prompt)
    classification = router_chain.run({"input": user_input})
    return classification.strip().lower()

def create_router_chain(llm, retriever, last_k_chat_history, user_input):
    """Combines RAG and generic chains with routing logic and returns the final answer."""
    # Create the history-aware RAG chain
    rag_chain = create_chains(llm, retriever)

    # Create the generic chain
    generic_chain = create_generic_prompt_chain(llm)
    
    # Determine whether input is 'specific' or 'generic'
    classification = router_logic(llm, user_input)

    if classification == "specific":
        # History-aware invocation for the RAG chain
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": last_k_chat_history
        })['answer']
    elif classification == "generic":
        # Direct invocation of the generic chain
        response = generic_chain.run({"input": user_input})
    else:
        # Fallback for unclassified cases
        response = "I'm sorry, I couldn't classify your input. Please try again."

    return response


def handle_chat(llm, retriever):
    """Handle user input and generate a response with chatbot-style UI."""
    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # User input field
    user_question = st.text_input("Enter your question:", key="user_input")

    if st.button("Submit"):
        if user_question.strip():
            # Use only the last k messages for response generation
            last_k_chat_history = get_last_k_messages(st.session_state["chat_history"], k=5)

            # Invoke the RAG chain
            # answer = rag_chain.invoke({
            #     "input": user_question,
            #     "chat_history": last_k_chat_history
            # })['answer']
            
            answer = create_router_chain(llm, retriever,last_k_chat_history,user_question)
            # Update chat history
            st.session_state["chat_history"].extend([
                {"role": "user", "content": user_question},
                {"role": "ai", "content": answer}
            ])

    # Display the entire chat history with styled messages
    st.subheader("Chat History")
    for message_item in st.session_state["chat_history"]:
        if message_item["role"] == "user":
            message(message_item["content"], is_user=True, avatar_style="person")
        elif message_item["role"] == "ai":
            message(message_item["content"], is_user=False, avatar_style="bot")
