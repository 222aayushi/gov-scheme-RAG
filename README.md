# gov-scheme-RAG
RAG-based conversational AI system for government welfare schemes with multilingual support, personalized retrieval, citation-backed responses, and hallucination reduction.

## Environment Setup

1. Recreate the virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. If PowerShell blocks activation, run one-time bypass:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
   . .venv\Scripts\Activate.ps1
   ```

3. Start the app:
   ```powershell
   .venv\Scripts\python.exe src\app.py
   ```

4. Open `http://127.0.0.1:5000/` in your browser.

## Notes

- If package installation fails because of network timeouts, retry with a longer timeout:
  ```powershell
  .venv\Scripts\python.exe -m pip install -r requirements.txt --timeout 120 --retries 10
  ```
- If `sentence-transformers` or `transformers` import hangs, recreate the `.venv` after updating `requirements.txt`.
- Make sure you have a valid API key in `.env` for `GENAI_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY` when using Gemini.
