from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
import transformers
import torch
import json
from flask_cors import CORS
from datetime import datetime
import asyncio
import os





# Initialize Flask application
app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={r"/": {"origins": ""}})

# Initialize Hugging Face model and tokenizer
model_name = "suzall/Llama-2-7b-chat-finetune-link-box"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Define system prompt for the medical summary generation
SYSTEM_PROMPT = """
   You are a professional medical assistant tasked with improving the accuracy and readability of symptom summaries. 
    You will receive a user-provided summary containing grammatical mistakes, vague details, and inaccuracies. 
    Your job is to recreate a concise, precise, and medically accurate summary.
    the output will be more detailed than the user summary, as we have mentioned below..

    For example:
    User Summary: Patient is a 42 year-old Male who presents with complaint of heartburn. The heartburn is located in the chest, and the throat. The heartburn is located in yes. 
    AI Summary: The patient is a 42-year-old male presenting with complaints of heartburn located in the chest and throat. Symptoms began 3 months ago, worsening over time...

    You will tailor your summary to each input provided by the user."""

def generate_llama_summary(user_input):
    """
    Generate a medical summary using the LLaMA model with a system prompt.
    
    Args:
        user_input (str): The user-provided medical summary text
        
    Returns:
        str: Generated improved medical summary
        
    Technical details:
    - Uses Hugging Face pipeline for text generation
    - Formats prompt with system instructions and user input
    - Configures generation parameters: top_k=10, max_length=200
    - Returns first generated sequence
    """
    formatted_prompt = f'[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_input} [/INST]'
    
    sequences = pipeline(
        formatted_prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    return sequences[0]['generated_text']

def save_to_jsonl(data):
    """
    Save feedback data to a JSONL file with timestamp.
    
    Args:
        data (dict): Dictionary containing user_summary, ai_summary, and rating
        
    Technical details:
    - Creates 'feedback' directory if it doesn't exist
    - Appends JSON records to feedback.jsonl
    - Each record includes timestamp and all feedback data
    - Uses ISO format for timestamp
    """
    os.makedirs('feedback', exist_ok=True)
    filename = 'feedback/feedback.jsonl'
    with open(filename, 'a') as f:
        json_record = {
            'timestamp': datetime.now().isoformat(),
            'user_summary': data['user_summary'],
            'ai_summary': data['ai_summary'],
            'rating': data['rating']
        }
        f.write(json.dumps(json_record) + '\n')

@app.route('/', methods=['GET'])
def home():
    """
    Render the main application page.
    
    Returns:
        HTML: Rendered index.html template
        
    Technical details:
    - Flask route for root URL
    - Uses Flask's render_template function
    """
    return render_template('index.html')

@app.route('/summary', methods=['GET'])
async def get_summary():
    """
    API endpoint to generate medical summaries.
    
    Query Parameters:
        text (str): Input text to summarize
        model (str): Model type to use (default: 'openai')
        
    Returns:
        JSON: Generated summary or error message
        
    Technical details:
    - Async route handler
    - Uses asyncio for asynchronous processing
    - Error handling for missing text and processing errors
    - Returns HTTP 400 for invalid requests, 500 for processing errors
    """
    text = request.args.get('text')
    model_type = request.args.get('model', 'openai')
    
    if not text:
        return jsonify({
            'status': 'error',
            'message': 'No text provided'
        }), 400
    
    try:
        summary = await asyncio.to_thread(generate_llama_summary, text)
        return jsonify({
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/ratings', methods=['POST'])
def save_feedback():
    """
    API endpoint to save user feedback on summaries.
    
    Request Body:
        user_summary (str): Original user summary
        ai_summary (str): Generated AI summary
        rating (float): User rating (0-5)
        
    Returns:
        JSON: Success or error message
        
    Technical details:
    - Validates required fields
    - Validates rating range (0-5)
    - Saves feedback to JSONL file
    - Returns HTTP 400 for validation errors, 500 for processing errors
    """
    data = request.json
    
    required_fields = ['user_summary', 'ai_summary', 'rating']
    if not all(field in data for field in required_fields):
        return jsonify({
            'status': 'error',
            'message': f'Missing required fields. Required: {required_fields}'
        }), 400
    
    try:
        rating = float(data['rating'])
        if not (0 <= rating <= 5):
            raise ValueError("Rating must be between 0 and 5")
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    
    try:
        save_to_jsonl(data)
        return jsonify({
            'status': 'success',
            'message': 'Rating saved successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)