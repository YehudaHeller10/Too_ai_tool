TOO Language Chat Assistant (local GGUF models)
==============================================

Screenshots
-----------
Below are example screenshots from the app showing a user typing a question and the model's streamed response.

User typing a question (user_req.png):

![User typing question](user_req.png)

Model response (AI_ans.png):

![Model answer](AI_ans.png)

If the images do not render in your viewer, ensure the files remain in the "OTHER VERSION/" folder next to the codebase.

Project layout
--------------
- TOO_AI_Assistant.py    -- Main PySide6 GUI application (in OTHER VERSION/)
- models/                -- Place your .gguf models here (examples included in repo)
- requirements.txt       -- Python dependencies

Features
--------
- Select and load any .gguf model from ./models via a dropdown.
- Streamed token-by-token generation to the UI.
- Simple TOO language knowledge base included as the context prompt.
- Marketplace item extractor (optional, requires network and requests).
- Basic message formatting (headers, code blocks, bold, headers).

Prerequisites
-------------
- Windows (tested), Python 3.10+
- Recommend a virtual environment

Install
-------
1. Create and activate a virtualenv:
   - python -m venv .venv
   - .\.venv\Scripts\activate

2. Install dependencies:
   - pip install -r requirements.txt

Notes about gpt4all and models
-----------------------------
- This project uses the gpt4all Python package to load and run local GGUF models.
- If you plan to use local GGUF models, ensure gpt4all and the runtime backend are installed correctly.
- Put any .gguf model files into the models/ directory (the GUI also searches ../models and the workspace recursively as a fallback).

Running the app
---------------
1. From the workspace root run (while virtualenv active):
   - python "OTHER VERSION\VER2.py"

2. In the GUI:
   - Select a model from the dropdown and click "Load Model".
   - When loaded, type a question about the TOO language and press Send (or Enter).
   - Responses stream token-by-token. Use Clear to reset the chat.

Troubleshooting
---------------
- "gpt4all package not found": pip install gpt4all
- No models found: place .gguf files into the models/ directory or ensure the path is correct.
- PySide6 errors: pip install -r requirements.txt (or pip install PySide6)
- If loading hangs or fails, check console output for stack traces and ensure the selected .gguf is compatible with your gpt4all runtime.

Privacy & License
-----------------
- Free for all uses


Authors
-------
- YEHUDA HELLER
- MOSHE NAMDAR
