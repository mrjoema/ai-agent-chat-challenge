# Thoughtful AI Customer Support Agent

A simple conversational AI agent that answers questions about Thoughtful AI's healthcare automation products.

## Features

- **Intelligent Question Matching**: Uses TF-IDF vectorization for finding relevant answers
- **LLM Fallback**: Uses OpenAI API for questions not in the knowledge base
- **Robust Error Handling**: Gracefully handles API failures, timeouts, and invalid inputs
- **Security**: Input validation and sanitization to prevent harmful inputs
- **User-Friendly Interface**: Clean Streamlit chat interface

## Quick Start

### 🚀 Easy Setup (Recommended)

1. **Run the startup script**:
   ```bash
   ./run.sh
   ```

   This script will automatically:
   - Create a virtual environment if needed
   - Install all dependencies
   - Copy `.env` template if needed
   - Start the Streamlit application

2. **Open your browser** to `http://localhost:8501`

3. **(Optional) Add OpenAI API key for enhanced responses**:
   - Edit the `.env` file and add your OpenAI API key
   - Restart the application

### 🔧 Manual Setup

If you prefer manual setup:

1. **Create virtual environment and install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key (optional)
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

- Ask questions about Thoughtful AI's products (EVA, CAM, PHIL)
- The agent will provide relevant answers or use AI-generated responses for other questions
- Works without OpenAI API key using hardcoded responses and keyword matching

## Example Questions

- "What does EVA do?"
- "Tell me about claims processing"
- "How does PHIL work?"
- "What are the benefits of using Thoughtful AI's agents?"

## Architecture

### Core Components

- **agent.py**: Core logic including input validation, question matching, and LLM integration
- **app.py**: Streamlit user interface with chat functionality
- **Multi-layer fallback**: TF-IDF → Keyword matching → LLM → Generic response

### Error Handling

The agent handles:
- Invalid or harmful inputs (XSS, SQL injection attempts)
- API timeouts and rate limits
- Connection failures
- Missing API keys
- Unexpected errors

All errors are logged and users receive helpful fallback messages.

### Security Features

- Input length validation
- Harmful pattern detection
- Content sanitization
- Rate limiting
- Response validation

## Configuration

Environment variables (optional):
- `OPENAI_API_KEY`: Your OpenAI API key for enhanced responses
- `LOG_LEVEL`: Logging level (default: INFO)
- `CONFIDENCE_THRESHOLD`: Matching confidence threshold (default: 0.3)
- `MAX_INPUT_LENGTH`: Maximum input length (default: 500)

## Knowledge Base

The agent currently knows about:
- **EVA**: Eligibility Verification Agent
- **CAM**: Claims Processing Agent
- **PHIL**: Payment Posting Agent
- General benefits of Thoughtful AI's automation solutions

## Technical Details

- **Framework**: Streamlit for web UI
- **NLP**: scikit-learn TF-IDF vectorization
- **Matching**: Cosine similarity with fallback to keyword matching
- **LLM**: OpenAI GPT-3.5-turbo for general questions
- **Error Handling**: Comprehensive retry logic and graceful degradation

## Development

The codebase demonstrates:
- Clean, modular architecture
- Comprehensive error handling
- Security best practices
- Professional UI/UX
- Easy deployment and configuration

Perfect for showcasing software engineering skills in a time-constrained environment!