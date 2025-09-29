# Claude Development Notes

## Important Reminders

### Server Management
- **Always restart the server after making changes to the code**
- The server has auto-reload enabled, but manual restarts ensure changes are properly applied
- Use: `python run.py` to start the server
- Server runs on http://localhost:8000

### Project Structure
- `main.py` - Core FastAPI application with simplified word-level diff algorithm
- `templates/index.html` - Frontend with dual prompt support
- `run.py` - Server startup script with enhanced reload configuration
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

### Key Features
- Dual prompt support (send different prompts to different models)
- Word-level diff visualization with interpolation
- Smart punctuation handling
- Bootstrap UI with custom diff styling
- OpenAI API integration

### Development Notes
- Current diff implementation uses simple word tokenization for better readability
- Avoids complex phrase grouping to prevent "rainbow effect"
- Allows natural word interpolation from both responses
- Maintains O(nm + L×k) algorithmic complexity