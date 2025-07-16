#!/usr/bin/env python3
"""
Simple test script to verify the agent works correctly
"""

import os
from agent import ConversationalAgent

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

def test_agent():
    """Test the agent with various inputs"""
    print("🤖 Testing Thoughtful AI Customer Support Agent")
    print("=" * 50)
    
    # Initialize agent
    qa_data = load_qa_data()
    agent = ConversationalAgent(qa_data)
    
    # Test cases
    test_cases = [
        "What does EVA do?",
        "Tell me about CAM",
        "How does PHIL work?",
        "What are the benefits?",
        "Hello there!",
        "What's the weather like?",
        "Invalid input: <script>alert('test')</script>",
        "",  # Empty input
        "a" * 600,  # Too long input
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. Input: '{test_input[:50]}{'...' if len(test_input) > 50 else ''}'")
        try:
            response = agent.get_response(test_input)
            print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        except Exception as e:
            print(f"   Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ Agent testing completed!")

if __name__ == "__main__":
    test_agent()