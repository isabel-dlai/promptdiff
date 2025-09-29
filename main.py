from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import difflib
import json
from typing import Optional

_ = load_dotenv()

# Load prompts from files
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory"""
    prompt_path = PROMPTS_DIR / filename
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

app = FastAPI(title="Response Diff Tool", description="Compare LLM responses visually")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class LLMProvider:
    def __init__(self):
        self.client = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create a single OpenAI client with proper connection limits"""
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                max_retries=2,
                timeout=30.0
            )
        return self.client

    async def get_response(self, prompt: str, model: str) -> str:
        if model.startswith("gpt"):
            return await self._openai_response(prompt, model)
        else:
            return f"Model {model} not implemented yet"

    async def _openai_response(self, prompt: str, model: str) -> str:
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    async def close(self):
        """Close the client connection"""
        if self.client:
            await self.client.close()
            self.client = None

llm_provider = LLMProvider()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/compare")
async def compare_responses(
    prompt1: str = Form(...),
    prompt2: Optional[str] = Form(None),
    model1: str = Form(...),
    model2: Optional[str] = Form(None)
):
    try:
        if model2 is None:
            model2 = model1

        if prompt2 is None or prompt2.strip() == "":
            prompt2 = prompt1

        import asyncio

        # Run both API calls concurrently to reduce latency
        response1, response2 = await asyncio.gather(
            llm_provider.get_response(prompt1, model1),
            llm_provider.get_response(prompt2, model2),
            return_exceptions=True
        )

        # Handle potential exceptions from gather
        if isinstance(response1, Exception):
            response1 = f"Error generating response 1: {str(response1)}"
        if isinstance(response2, Exception):
            response2 = f"Error generating response 2: {str(response2)}"

        # Generate AI summary of differences
        try:
            summary = await generate_difference_summary(response1, response2, prompt1, prompt2)
        except Exception as e:
            summary = "Summary generation failed"

        diff_html = generate_diff_html(response1, response2)

        return {
            "prompt1": prompt1,
            "prompt2": prompt2,
            "model1": model1,
            "model2": model2,
            "response1": response1,
            "response2": response2,
            "summary": summary,
            "diff_html": diff_html
        }
    except Exception as e:
        # Reset the client on error to ensure clean state
        await llm_provider.close()
        raise e

async def generate_difference_summary(response1: str, response2: str, prompt1: str, prompt2: str) -> str:
    """Generate a pithy AI summary of substantive differences between responses"""

    # Load prompt template and format with responses
    prompt_template = load_prompt("comparison_summary.md")
    analysis_prompt = prompt_template.format(response1=response1, response2=response2)

    try:
        client = llm_provider._get_client()
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use faster model for analysis
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=100,  # Keep it brief
            temperature=0.3  # Lower temperature for consistent analysis
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable: {str(e)}"

def generate_diff_html(text1: str, text2: str) -> str:
    import re
    from difflib import SequenceMatcher
    import html

    # Enhanced word tokenization that separates punctuation
    def tokenize_words(text):
        """Split text into words and punctuation, preserving whitespace"""
        # Match: word characters, punctuation, or whitespace
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
        return tokens

    # Split texts into lines for line-by-line comparison
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    matcher = SequenceMatcher(None, lines1, lines2)
    diff_html = '<div class="custom-diff">'

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Show unchanged lines
            for i in range(i1, i2):
                line = html.escape(lines1[i]) if lines1[i] else '&nbsp;'
                diff_html += f'<div class="diff-equal"><span class="diff-marker">&nbsp;</span><span class="line-content">{line}</span></div>'
        elif tag == 'delete':
            # Show deleted lines
            for i in range(i1, i2):
                line = html.escape(lines1[i]) if lines1[i] else '&nbsp;'
                diff_html += f'<div class="diff-delete"><span class="diff-marker">-</span><span class="line-content">{line}</span></div>'
        elif tag == 'insert':
            # Show inserted lines
            for j in range(j1, j2):
                line = html.escape(lines2[j]) if lines2[j] else '&nbsp;'
                diff_html += f'<div class="diff-insert"><span class="diff-marker">+</span><span class="line-content">{line}</span></div>'
        elif tag == 'replace':
            # Show word-level differences for replaced lines
            old_lines = lines1[i1:i2]
            new_lines = lines2[j1:j2]

            if len(old_lines) == len(new_lines) == 1:
                # Single line replacement - show word-level diff with interpolation
                old_text = old_lines[0]
                new_text = new_lines[0]

                # Tokenize into words
                old_words = tokenize_words(old_text)
                new_words = tokenize_words(new_text)

                # Helper function to check if tokens are similar punctuation
                def is_similar_punctuation(token1, token2):
                    """Check if two tokens are similar punctuation (e.g., ':' vs '-')"""
                    punct_chars = set('.,;:!?-—–()[]{}"/\\|~`^*+=<>')
                    return (len(token1) == 1 and len(token2) == 1 and
                            token1 in punct_chars and token2 in punct_chars)

                # Compare word by word with enhanced punctuation handling
                word_matcher = SequenceMatcher(None, old_words, new_words)
                unified_line = ""

                for word_tag, wi1, wi2, wj1, wj2 in word_matcher.get_opcodes():
                    if word_tag == 'equal':
                        # Common words - show normally
                        unified_line += html.escape(''.join(old_words[wi1:wi2]))
                    elif word_tag == 'delete':
                        # Deleted words - show with strikethrough
                        deleted_text = ''.join(old_words[wi1:wi2])
                        unified_line += f'<span class="char-delete">{html.escape(deleted_text)}</span>'
                    elif word_tag == 'insert':
                        # Inserted words - show with highlight
                        inserted_text = ''.join(new_words[wj1:wj2])
                        unified_line += f'<span class="char-insert">{html.escape(inserted_text)}</span>'
                    elif word_tag == 'replace':
                        # Check for similar punctuation replacements
                        old_segment = old_words[wi1:wi2]
                        new_segment = new_words[wj1:wj2]

                        # If it's a single token replacement and both are similar punctuation
                        if (len(old_segment) == 1 and len(new_segment) == 1 and
                            is_similar_punctuation(old_segment[0], new_segment[0])):
                            # Show as a minor change rather than delete+insert
                            unified_line += f'<span class="char-delete">{html.escape(old_segment[0])}</span>'
                            unified_line += f'<span class="char-insert">{html.escape(new_segment[0])}</span>'
                        else:
                            # Standard replacement - show both delete and insert
                            deleted_text = ''.join(old_segment)
                            inserted_text = ''.join(new_segment)
                            unified_line += f'<span class="char-delete">{html.escape(deleted_text)}</span>'
                            unified_line += f'<span class="char-insert">{html.escape(inserted_text)}</span>'

                diff_html += f'<div class="diff-changed"><span class="diff-marker">~</span><span class="line-content">{unified_line}</span></div>'
            else:
                # Multiple lines - show as separate delete/insert blocks
                for i in range(len(old_lines)):
                    line = html.escape(old_lines[i]) if old_lines[i] else '&nbsp;'
                    diff_html += f'<div class="diff-delete"><span class="diff-marker">-</span><span class="line-content">{line}</span></div>'
                for j in range(len(new_lines)):
                    line = html.escape(new_lines[j]) if new_lines[j] else '&nbsp;'
                    diff_html += f'<div class="diff-insert"><span class="diff-marker">+</span><span class="line-content">{line}</span></div>'

    diff_html += '</div>'
    return diff_html
