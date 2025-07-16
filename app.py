import streamlit as st
import time
import json
import logging
from agent import ConversationalAgent

# Configure page
st.set_page_config(
    page_title="Thoughtful AI Support",
    page_icon="🤖",
    layout="centered"
)

# Load Q&A data
def load_qa_data():
    """Load the predefined Q&A dataset"""
    return [
        {
            "question": "What does the eligibility verification agent (EVA) do?",
            "answer": "EVA automates the process of verifying a patient's eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
        },
        {
            "question": "What does the claims processing agent (CAM) do?",
            "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
        },
        {
            "question": "How does the payment posting agent (PHIL) work?",
            "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
        },
        {
            "question": "Tell me about Thoughtful AI's Agents.",
            "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
        },
        {
            "question": "What are the benefits of using Thoughtful AI's agents?",
            "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
        }
    ]

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm the Thoughtful AI support agent. I can help you learn about our healthcare automation solutions including EVA, CAM, and PHIL. How can I help you today?"
        })
    
    if 'agent' not in st.session_state:
        with st.spinner("Initializing agent..."):
            try:
                qa_data = load_qa_data()
                st.session_state.agent = ConversationalAgent(qa_data)
            except Exception as e:
                st.error("Failed to initialize the agent. Please refresh the page.")
                logging.error(f"Initialization error: {str(e)}")
                st.stop()
    
    if 'last_message_time' not in st.session_state:
        st.session_state.last_message_time = 0

def main():
    """Main application function"""
    st.title("🤖 Thoughtful AI Support Agent")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask about Thoughtful AI's automation agents..."):
        # Rate limiting
        current_time = time.time()
        if current_time - st.session_state.last_message_time < 1:
            st.warning("Please wait a moment before sending another message.")
            return
        
        st.session_state.last_message_time = current_time
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.get_response(prompt)
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error("An unexpected error occurred. Please try again.")
            logging.error(f"Chat error: {str(e)}")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### About Thoughtful AI")
        st.info(
            "Thoughtful AI provides intelligent automation solutions for healthcare:\n\n"
            "- **EVA**: Eligibility Verification\n"
            "- **CAM**: Claims Processing\n"
            "- **PHIL**: Payment Posting"
        )
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Chat cleared! How can I help you learn about Thoughtful AI today?"
            })
            st.rerun()

if __name__ == "__main__":
    main()