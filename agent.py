import re
import logging
import time
import os
from typing import Optional, Tuple, List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InputValidator:
    """Validates and sanitizes user input"""
    
    def __init__(self):
        self.max_input_length = 500
        self.min_input_length = 2
        self.harmful_patterns = [
            r'<script.*?>.*?</script>',
            r'DROP\s+TABLE',
            r'exec\s*\(',
            r'__import__',
            r'eval\s*\(',
            r'javascript:',
            r'onerror\s*=',
        ]
    
    def validate_input(self, user_input: str) -> Tuple[bool, str]:
        """Validate and sanitize user input"""
        try:
            if not user_input or not isinstance(user_input, str):
                return False, "Please enter a valid question."
            
            cleaned_input = user_input.strip()
            
            if len(cleaned_input) < self.min_input_length:
                return False, "Your question is too short. Please provide more detail."
            
            if len(cleaned_input) > self.max_input_length:
                return False, f"Your question is too long. Please keep it under {self.max_input_length} characters."
            
            # Check for harmful patterns
            for pattern in self.harmful_patterns:
                if re.search(pattern, cleaned_input, re.IGNORECASE):
                    logging.warning(f"Potentially harmful input detected: {pattern}")
                    return False, "Invalid input detected. Please rephrase your question."
            
            # Clean the input
            cleaned_input = re.sub(r'\s+', ' ', cleaned_input)
            cleaned_input = re.sub(r'[<>]', '', cleaned_input)  # Remove angle brackets
            
            return True, cleaned_input
            
        except Exception as e:
            logging.error(f"Input validation error: {str(e)}")
            return False, "An error occurred while processing your input. Please try again."


class QuestionMatcher:
    """Handles question matching using TF-IDF vectorization"""
    
    def __init__(self, qa_data: List[Dict]):
        self.qa_data = qa_data
        self.questions = [item['question'] for item in qa_data]
        self.answers = [item['answer'] for item in qa_data]
        self.vectorizer = None
        self.question_vectors = None
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer"""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            self.question_vectors = self.vectorizer.fit_transform(self.questions)
            logging.info("Vectorizer initialized successfully")
        except Exception as e:
            logging.error(f"Vectorizer initialization failed: {str(e)}")
            self.vectorizer = None
    
    def find_best_match(self, user_input: str, threshold: float = 0.3) -> Tuple[Optional[str], float]:
        """Find the best matching answer for user input"""
        try:
            if self.vectorizer and self.question_vectors is not None:
                # Vector-based matching
                user_vector = self.vectorizer.transform([user_input.lower()])
                similarities = cosine_similarity(user_vector, self.question_vectors)[0]
                
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                if best_score >= threshold:
                    return self.answers[best_idx], best_score
                else:
                    return None, best_score
            else:
                # Fallback to keyword matching
                return self._keyword_match(user_input, threshold)
                
        except Exception as e:
            logging.error(f"Matching error: {str(e)}")
            return None, 0.0
    
    def _keyword_match(self, user_input: str, threshold: float) -> Tuple[Optional[str], float]:
        """Simple keyword-based matching as fallback"""
        user_words = set(user_input.lower().split())
        best_score = 0.0
        best_answer = None
        
        for i, question in enumerate(self.questions):
            question_words = set(question.lower().split())
            
            # Calculate Jaccard similarity
            intersection = user_words.intersection(question_words)
            union = user_words.union(question_words)
            score = len(intersection) / len(union) if union else 0
            
            if score > best_score:
                best_score = score
                best_answer = self.answers[i]
        
        if best_score >= threshold:
            return best_answer, best_score
        return None, best_score


class ConversationalAgent:
    """Main conversational agent with error handling and fallback"""
    
    def __init__(self, qa_data: List[Dict]):
        self.validator = InputValidator()
        self.matcher = QuestionMatcher(qa_data)
        self.openai_client = self._initialize_openai()
        self.max_retries = 3
        self.retry_delay = 1
    
    def _initialize_openai(self) -> Optional[openai.OpenAI]:
        """Initialize OpenAI client with error handling"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logging.warning("OpenAI API key not found. Fallback responses will be limited.")
                return None
            
            client = openai.OpenAI(api_key=api_key)
            # Test the connection
            client.models.list()
            logging.info("OpenAI client initialized successfully")
            return client
            
        except openai.AuthenticationError:
            logging.error("Invalid OpenAI API key")
            return None
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {str(e)}")
            return None
    
    def get_response(self, user_input: str) -> str:
        """Get response for user input with comprehensive error handling"""
        try:
            # Validate input
            is_valid, cleaned_input = self.validator.validate_input(user_input)
            if not is_valid:
                return cleaned_input  # Return error message
            
            # Try to find matching answer
            answer, confidence = self.matcher.find_best_match(cleaned_input)
            
            if answer and confidence > 0.6:
                # High confidence match
                return answer
            elif answer and confidence > 0.3:
                # Medium confidence - add context
                return f"{answer}\n\nIf this doesn't fully answer your question, feel free to ask it differently!"
            else:
                # No good match - use LLM fallback
                return self._get_fallback_response(cleaned_input)
                
        except Exception as e:
            logging.error(f"Critical error in get_response: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
    
    def _get_fallback_response(self, user_input: str) -> str:
        """Get fallback response using OpenAI API"""
        try:
            if self.openai_client:
                return self._get_llm_response_with_retry(user_input)
            else:
                return (
                    "I don't have specific information about that in my knowledge base. "
                    "Please ask about our healthcare automation agents: "
                    "EVA (eligibility verification), CAM (claims processing), or PHIL (payment posting)."
                )
        except Exception as e:
            logging.error(f"Fallback response error: {str(e)}")
            return "I'm having trouble processing your question. Could you please rephrase it?"
    
    def _get_llm_response_with_retry(self, user_input: str) -> str:
        """Get LLM response with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_openai_api(user_input)
                if response:
                    return self._validate_llm_response(response)
                    
            except openai.RateLimitError as e:
                last_error = e
                wait_time = self.retry_delay * (2 ** attempt)
                logging.warning(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except openai.APITimeoutError as e:
                last_error = e
                logging.warning(f"API timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except openai.APIConnectionError as e:
                last_error = e
                logging.error(f"Connection error: {str(e)}")
                return self._get_connection_error_response()
                
            except Exception as e:
                logging.error(f"Unexpected error calling OpenAI: {str(e)}")
                last_error = e
                break
        
        return self._get_final_fallback_response(last_error)
    
    def _call_openai_api(self, user_input: str) -> Optional[str]:
        """Make OpenAI API call"""
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a helpful customer support agent for Thoughtful AI. "
                        "Be concise and friendly. If asked about specific products, "
                        "mention that you can provide detailed information about "
                        "EVA, CAM, and PHIL if they ask specifically."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,
            temperature=0.7,
            timeout=10.0
        )
        
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        return None
    
    def _validate_llm_response(self, response: str) -> str:
        """Validate and sanitize LLM response"""
        if not response or not response.strip():
            return "I couldn't generate a proper response. Please try asking your question differently."
        
        # Truncate if too long
        if len(response) > 1000:
            response = response[:1000] + "..."
        
        # Check for harmful content
        harmful_patterns = ['<script', 'javascript:', 'onclick=', 'onerror=']
        response_lower = response.lower()
        for pattern in harmful_patterns:
            if pattern in response_lower:
                logging.warning(f"Potentially harmful content in LLM response: {pattern}")
                return "I encountered an issue generating a safe response. Please try again."
        
        return response.strip()
    
    def _get_connection_error_response(self) -> str:
        """Response when API is unreachable"""
        return (
            "I'm currently unable to connect to my full knowledge base. "
            "However, I can tell you about Thoughtful AI's main automation agents: "
            "EVA for eligibility verification, CAM for claims processing, "
            "and PHIL for payment posting. Please ask about any of these!"
        )
    
    def _get_final_fallback_response(self, error: Exception) -> str:
        """Final fallback when all retries fail"""
        if isinstance(error, openai.RateLimitError):
            return (
                "I'm experiencing high demand right now. "
                "In the meantime, feel free to ask about our specific agents: "
                "EVA, CAM, or PHIL, and I can provide detailed information about them."
            )
        else:
            return (
                "I apologize, but I'm having technical difficulties with that question. "
                "I can still provide information about our automation agents: "
                "EVA (eligibility), CAM (claims), and PHIL (payments)."
            )