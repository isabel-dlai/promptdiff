# Response Diff Tool

A web application built with FastAPI that allows you to send the same prompt to different LLM models and compare their responses with a visual diff.

## Features

- Send prompts to multiple LLM models
- Visual diff comparison of responses
- Support for OpenAI models (GPT-3.5, GPT-4, etc.)
- Clean web interface with Bootstrap styling
- Side-by-side response comparison

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

3. Run the application:
```bash
python run.py
```

4. Open your browser to `http://localhost:8000`

## Usage

1. Enter your prompt in the text area
2. Select the first model from the dropdown
3. Optionally select a second model (if left empty, uses the same model twice)
4. Click "Compare Responses" to see the results and visual diff

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key (for future implementation)

## Extending

To add support for additional LLM providers, modify the `LLMProvider` class in `main.py` and add the corresponding API integration.
