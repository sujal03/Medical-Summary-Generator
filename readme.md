# Medical Summary Generator

A Flask application that generates medical summaries using the LLaMA model.

## Features

* Generates medical summaries from user-provided text
* Saves user feedback on summaries to a JSONL file
* Supports multiple models (currently only OpenAI)

## Requirements

* Python 3.8+
* Flask 2.0.2
* Flask-Cors 3.0.10
* Transformers 4.17.0
* Torch 1.10.0
* OpenAI 0.10.0
* Python-dotenv 0.19.2

## Installation

1. Clone the repository: `git clone https://github.com/your-username/medical-summary-generator.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (on Linux/Mac) or `venv\Scripts\activate` (on Windows)
4. Install the requirements: `pip install -r requirements.txt`
5. Create a `.env` file with your OpenAI API key: `OPENAI_API_KEY=your-api-key`

## Running the Application

1. Run the application: `python app.py`
2. Open a web browser and navigate to `http://localhost:5000`

## API Endpoints

* `/`: Render the main application page
* `/summary`: Generate a medical summary from user-provided text
* `/ratings`: Save user feedback on summaries

## Example Use Cases

* Generate a medical summary: `curl -X GET 'http://localhost:5000/summary?text=Your+medical+summary+text'`
* Save user feedback: `curl -X POST -H 'Content-Type: application/json' -d '{"user_summary": "Your user summary", "ai_summary": "Your AI summary", "rating": 4}' http://localhost:5000/ratings`