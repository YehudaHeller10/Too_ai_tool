"""
TOO Language Chat Assistant (using local GGUF models with streaming)
---------------------------------------------------------------------
- Adds a dropdown to select and load any GGUF model from the ./models directory.
- Streams generation tokens live to the UI (token-by-token).
- Uses the provided TOO "Core Principles" text as the knowledge/context prompt.
- Attempts to fetch marketplace items from https://too.software/marketplace/ (optional).
- MODIFIED: Adds a loading spinner and markdown-style formatting for responses.
"""

import sys
import os
import threading
import traceback
from pathlib import Path
from typing import List, Optional
import time
import html
import re

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QTextEdit, QLineEdit, QPushButton, QListWidget, QStatusBar, QMessageBox,
    QComboBox
)
from PySide6.QtCore import Qt, QTimer, QTime, Slot, QObject, Signal
from PySide6.QtGui import QTextCursor

# Optional network fetching for marketplace
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# GPT4All import
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except Exception as e:
    print("Warning: gpt4all package not found. Install with: pip install gpt4all")
    GPT4ALL_AVAILABLE = False


# -------------------------
# The TOO knowledge source
# -------------------------
TOO_KNOWLEDGE = r"""
Core Principles

Everything is a thing – booleans, numbers, sensors, console, SMS, AI… all are things.

Rule-based – logic is expressed with when ... { actions }.

Maps instead of arrays – elements are accessed by names, not numeric indices.

Functional – functions may return multiple values, be variadic, or anonymous.

No variables, primitive types, loops, or binary operators.

Yes lightweight threads, channels, inheritance, interfaces.

Syntax (12 rules)
thing ::= Id "{" (declare|signal|when|func)* "}"

signal ::= Id params

when ::= source+ Id params "{" action* "}"

func ::= Id params "→" params ["{" action* "}"]

action ::= ["*"] expr ["?" "{" action* "}" ["{" action* "}"]]

source ::= Id ["[" declare "]"]

expr ::= sub+
sub ::= (Id|Const|call) ["[" arg "]"] ["→" (declare|params)]
call ::= Id "(" [(arg ",")* arg] ")"
params ::= "(" [(declare ",")* (declare ["..."])] ")"
declare ::= ["[]"] [Id ":"] (Id ".")* Id ["=" Const] [Const]
arg ::= expr|"*"

5 Operators

→ Redirect (assignment, right-to-left).

[] Subscript (maps).

? Conditional.

* Iteration / wildcard.

= Initial value in declaration.

Canonical Examples
Hello world
hello {
  console : console
  when hello {
    console print("Hello world!")
  }
}

Summation with map
sigma {
  set : [] number
  when sigma {
    sum = 0
    set[*] number → x
    sum plus(x) → sum
    console print(sum)
  }
}

Variadic + interface
hi {
  when hi {
    terminal print("SPLASH", hi, 23)
  }
  func as string() → (string) {
    "Cascais"
  }
}

Broadcast key/value
broadcast {
  phone : [] string
  when broadcast {
    phone[*] → m
    terminal print("Hello", *, "greetings from Cascais!")
    sms send(m, "Hello " + * + "!")
  }
}

Fibonacci (conditional loop)
fibonacci {
  n : number = 10
  when fibonacci {
    * (n greater(0))? {
      console print(f(n))
      n minus(1) → n
    }
  }
  func f(n:number) → (number) {
    ((phi power(n)) minus(psi power(n))) divided·by(sqrt(5))
  }
}

Key Usage Principles

No for/while. Use * (iteration) and ? (decision).

Natural-style chaining:
radius squared times π → area times height → volume.

Concurrency with lightweight threads:
go run(task[*], *).
"""

# -------------------------
# Marketplace helper
# -------------------------
def fetch_marketplace_items() -> List[str]:
    """Try to fetch marketplace items from too.software; fallback to static list."""
    fallback = [
        "boolean", "duration", "integer", "location", "number", "string",
        "tick", "error", "regex", "terminal", "time", "timer",
        "list", "dlist", "queue", "stack", "hash", "heap",
        "http", "sms", "mailer"
    ]
    url = "https://too.software/marketplace/"
    if not REQUESTS_AVAILABLE:
        return fallback
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return fallback
        text = r.text
        import re
        candidates = re.findall(r"[A-Za-z0-9·\-\_]{3,40}", text)
        seen = set()
        names = []
        for c in candidates:
            cl = c.strip()
            if cl.lower() in seen: continue
            if len(cl) > 30 or cl.isdigit(): continue
            seen.add(cl.lower())
            names.append(cl)
            if len(names) >= 60: break
        if names: return names
    except Exception:
        pass
    return fallback


# -------------------------
# LLM / GPT4All handler
# -------------------------
class LLMHandler(QObject):
    response_ready = Signal(str)
    status_update = Signal(str)
    generation_finished = Signal()

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_loaded = False
        self._stream_lock = threading.Lock()

    @staticmethod
    def find_all_gguf_models() -> List[str]:
        """Find all .gguf model files in standard directories."""
        models = []
        for mdl_dir in [Path("models"), Path("../models")]:
            if mdl_dir.exists():
                models.extend([str(p.resolve()) for p in mdl_dir.rglob("*.gguf")])
        # Also check current directory as a fallback
        if not models:
            models.extend([str(p.resolve()) for p in Path(".").rglob("*.gguf")])
        return sorted(list(set(models))) # Return unique, sorted list of paths

    def load_model_by_path(self, model_path: str):
        """Starts a background thread to load the specified model."""
        if not model_path:
            self.status_update.emit("No model path provided.")
            return
        threading.Thread(target=self._load_model_worker, args=(model_path,), daemon=True).start()

    def _load_model_worker(self, model_path: str):
        """The actual model loading logic that runs in a thread."""
        if not GPT4ALL_AVAILABLE:
            self.status_update.emit("gpt4all package not installed (pip install gpt4all)")
            return

        if not Path(model_path).exists():
            self.status_update.emit(f"Model file not found: {model_path}")
            return

        self.status_update.emit(f"Loading model: {Path(model_path).name}")
        self.model_loaded = False
        self.model = None # Release old model if any

        try:
            # GPT4All handles resource management internally when a new model is loaded.
            m = GPT4All(model_path)
            self.model = m
            self.model_loaded = True
            self.status_update.emit(f"Model loaded successfully: {Path(model_path).name}")
        except Exception as e:
            msg = f"Failed to construct GPT4All model: {e}"
            self.status_update.emit(msg)
            traceback.print_exc()

    def _create_too_prompt(self, user_message: str) -> str:
        if not user_message or not user_message.strip():
            user_message = "Hello, please introduce TOO language"
        prompt = (
            "You are an expert assistant for the TOO programming language (Things Object-Oriented).\n\n"
            "Base knowledge (do not invent contradictions):\n"
            f"{TOO_KNOWLEDGE}\n\n"
            "User message / context:\n"
            f"{user_message}\n\n"
            "Instructions:\n"
            "- Use the Base knowledge above as the authoritative source when answering.\n"
            "- If the user asks for code examples, provide short, correct examples in TOO syntax.\n"
            "- If unsure, state uncertainty and avoid inventing specifics not in the Base knowledge.\n"
            "- Keep answers focused and practical for a programmer learning or using TOO.\n"
            "- Give Code Example in Too language if relevant.\n"
        )
        return prompt

    def process_message(self, message: str):
        if not self.model_loaded or not self.model:
            self.response_ready.emit("Model not ready. Please load a model first.")
            self.generation_finished.emit()
            return
        threading.Thread(target=self._generation_worker, args=(message,), daemon=True).start()

    def _generation_worker(self, msg: str):
        if not self._stream_lock.acquire(blocking=False):
            self.response_ready.emit("Another generation is already in progress.")
            self.generation_finished.emit()
            return

        try:
            self.status_update.emit("Generating response (streaming)...")
            prompt = self._create_too_prompt(msg)
            partial_tokens = []

            for token in self.model.generate(prompt, max_tokens=800, temp=0.7, streaming=True):
                partial_tokens.append(token)
                self.response_ready.emit("".join(partial_tokens))

            self.status_update.emit("Idle")
        except Exception as e:
            self.response_ready.emit(f"Error during generation: {e}")
            self.status_update.emit("Error")
        finally:
            self.generation_finished.emit()
            self._stream_lock.release()


# -------------------------
# GUI Application
# -------------------------
class TOOChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.llm_handler = LLMHandler()
        self.marketplace_items = fetch_marketplace_items()
        self.model_map = {}
        self._assistant_response_start_pos = -1

        # --- Spinner setup ---
        self.spinner_timer = QTimer(self)
        self.spinner_chars = ["/", "-", "\\", "|"]
        self.spinner_index = 0

        self.setup_ui()
        self.connect_signals()
        self.populate_model_dropdown()

    def setup_ui(self):
        self.setWindowTitle("TOO Language Chat Assistant (local GGUF, streaming)")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Chat frame (left)
        chat_frame = QFrame()
        chat_frame.setFrameStyle(QFrame.StyledPanel)
        chat_layout = QVBoxLayout(chat_frame)

        # --- Model Selection Area ---
        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("Select Model:"))
        self.model_selector = QComboBox()
        self.model_selector.setToolTip("Select a .gguf model from the 'models' directory.")
        model_selection_layout.addWidget(self.model_selector, 1) # Stretch dropdown
        self.load_model_button = QPushButton("Load Model")
        model_selection_layout.addWidget(self.load_model_button)
        chat_layout.addLayout(model_selection_layout)
        # --- End Model Selection Area ---

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask about TOO language or paste TOO code...")
        self.send_button = QPushButton("Send")
        self.send_button.setEnabled(False) # Disabled until model is loaded
        self.clear_button = QPushButton("Clear")
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.clear_button)
        chat_layout.addLayout(input_layout)

        # Marketplace frame (right)
        marketplace_frame = QFrame()
        marketplace_frame.setFrameStyle(QFrame.StyledPanel)
        marketplace_layout = QVBoxLayout(marketplace_frame)
        marketplace_layout.addWidget(QLabel("🏪 TOO Marketplace (live if reachable)"))
        self.items_list = QListWidget()
        self.items_list.addItems(self.marketplace_items)
        marketplace_layout.addWidget(self.items_list)
        self.description_label = QLabel("Select an item to see description.")
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignTop)
        marketplace_layout.addWidget(self.description_label, 1)

        main_layout.addWidget(chat_frame, 7)
        main_layout.addWidget(marketplace_frame, 3)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Please select and load a model to begin.")

        self._append_system("🚀 Welcome! Please select a model from the dropdown and click 'Load Model'.")

    def populate_model_dropdown(self):
        """Finds all GGUF models and populates the QComboBox."""
        self.model_selector.clear()
        available_models = self.llm_handler.find_all_gguf_models()
        if available_models:
            self.model_map = {Path(p).name: p for p in available_models}
            self.model_selector.addItems(self.model_map.keys())
            self.load_model_button.setEnabled(True)
        else:
            self.status_bar.showMessage("No .gguf models found in the 'models' directory.")
            self.model_selector.setEnabled(False)
            self.load_model_button.setEnabled(False)

    def connect_signals(self):
        self.load_model_button.clicked.connect(self.on_load_model_clicked)
        self.send_button.clicked.connect(self.send_message)
        self.message_input.returnPressed.connect(self.send_message)
        self.clear_button.clicked.connect(self.clear_chat)
        self.items_list.itemClicked.connect(self.on_item_clicked)
        # LLM signals
        self.llm_handler.response_ready.connect(self.handle_llm_response)
        self.llm_handler.status_update.connect(self.update_status)
        self.llm_handler.generation_finished.connect(self.on_generation_finished)
        # Spinner timer
        self.spinner_timer.timeout.connect(self._update_spinner_animation)

    def _append_assistant_header(self):
        timestamp = QTime.currentTime().toString("hh:mm:ss")
        header = f"[{timestamp}] TOO Assistant:"
        html_header = f"<div style='margin-top:6px; color:#009900;font-weight:bold;'>{header}</div>"
        self.chat_display.append(html_header)
        self.chat_display.moveCursor(QTextCursor.End)

    def _append_message(self, sender: str, message: str):
        timestamp = QTime.currentTime().toString("hh:mm:ss")
        color = "#0066cc" if sender == "User" else "#666666"
        formatted_message = self._format_text_to_html(message)
        html_message = (
            f"<div style='margin:6px 0;'><span style='color:{color};font-weight:bold;'>"
            f"[{timestamp}] {sender}:</span><br><div style='margin-left:12px;'>{formatted_message}</div></div>"
        )
        self.chat_display.append(html_message)
        self.chat_display.moveCursor(QTextCursor.End)
        self.chat_display.ensureCursorVisible()

    def _append_system(self, text: str):
        self._append_message("System", text)

    def _append_user(self, text: str):
        timestamp = QTime.currentTime().toString("hh:mm:ss")
        sender_name = "User"
        # Escape user's input to avoid HTML injection and misinterpretation
        escaped_text = html.escape(text)
        # Replace newlines for correct rendering
        formatted_html = escaped_text.replace('\n', '<br>')

        html_message = (
            f"<div style='margin:6px 0;'><span style='color:#0066cc;font-weight:bold;'>"
            f"[{timestamp}] {sender_name}:</span><br><div style='margin-left:12px;white-space:pre-wrap;'>"
            f"{formatted_html}"
            f"</div></div>"
        )
        self.chat_display.append(html_message)
        self.chat_display.moveCursor(QTextCursor.End)
        self.chat_display.ensureCursorVisible()

    @staticmethod
    def _escape_html(text: str) -> str:
        return html.escape(text).replace("\n", "<br>")

    def _format_text_to_html(self, raw_text: str) -> str:
        """
        Converts text with markdown-like syntax (**, ###, ```, and now ####) to HTML.
        Removed background and border for code blocks.
        """
        parts = re.split(r'(```[\s\S]*?```)', raw_text)
        formatted_parts = []

        for part in parts:
            if not part: continue
            if part.startswith('```'):
                # Code block - removed background-color and border
                code_content = part[3:-3].strip()
                code_content = re.sub(r'^\w+\n', '', code_content, count=1)
                escaped_code = html.escape(code_content)
                formatted_code = (
                    f'<div style="margin: 8px 0; color: #333333; ' # Changed color for code to be visible on white background
                    f'padding: 10px; border-radius: 5px; font-family: Consolas, monospace; white-space: pre;">'
                    f'{escaped_code}'
                    f'</div>'
                )
                formatted_parts.append(formatted_code)
            else:
                # Regular prose
                escaped_prose = html.escape(part)
                formatted_prose = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped_prose)
                # Correctly handle headers, starting with ####
                formatted_prose = re.sub(r'(?m)^#### (.*)', r'<h4 style="margin-top:10px; margin-bottom:0;">\1</h4>', formatted_prose)
                formatted_prose = re.sub(r'(?m)^### (.*)', r'<h3 style="margin-top:10px; margin-bottom:0;">\1</h3>', formatted_prose)
                formatted_prose = formatted_prose.replace('\n', '<br>')
                formatted_parts.append(formatted_prose)

        return "".join(formatted_parts)

    # --- UI Slots ---

    @Slot()
    def _update_spinner_animation(self):
        """Updates the spinner character in the chat display."""
        cursor = QTextCursor(self.chat_display.document())
        cursor.setPosition(self._assistant_response_start_pos)
        cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 1)
        cursor.removeSelectedText()
        char = self.spinner_chars[self.spinner_index]
        cursor.insertText(char)
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)

    @Slot()
    def on_load_model_clicked(self):
        selected_model_name = self.model_selector.currentText()
        if not selected_model_name:
            QMessageBox.warning(self, "No Model Selected", "Please select a model from the dropdown.")
            return

        model_path = self.model_map.get(selected_model_name)
        self.send_button.setEnabled(False)
        self.load_model_button.setEnabled(False)
        self.model_selector.setEnabled(False)
        self._append_system(f"Attempting to load model: {selected_model_name}...")
        self.llm_handler.load_model_by_path(model_path)

    @Slot()
    def send_message(self):
        message = self.message_input.text().strip()
        if not message:
            return
        self._append_user(message)
        self.message_input.clear()
        self.send_button.setEnabled(False)

        self._append_assistant_header()
        self._assistant_response_start_pos = self.chat_display.textCursor().position()
        self.chat_display.insertPlainText(" ")
        self.spinner_timer.start(150)

        self.llm_handler.process_message(message)

    @Slot()
    def clear_chat(self):
        self.chat_display.clear()
        self._append_system("Chat cleared. Ask about TOO.")
        self._assistant_response_start_pos = -1

    @Slot()
    def on_item_clicked(self, item):
        name = item.text()
        desc = f"Marketplace item: {name} (fetched live if available)."
        self.description_label.setText(desc)

    @Slot(str)
    def handle_llm_response(self, response: str):
        self.spinner_timer.stop()
        if self._assistant_response_start_pos < 0: return

        cursor = self.chat_display.textCursor()
        cursor.setPosition(self._assistant_response_start_pos)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()

        formatted_html = self._format_text_to_html(response)
        cursor.insertHtml(f"<div style='margin-left:12px;'>{formatted_html}</div>")
        self.chat_display.ensureCursorVisible()

    @Slot()
    def on_generation_finished(self):
        self.spinner_timer.stop()
        if self.llm_handler.model_loaded:
            self.send_button.setEnabled(True)
        self._assistant_response_start_pos = -1

    @Slot(str)
    def update_status(self, status: str):
        self.status_bar.showMessage(status)
        if "Model loaded successfully" in status:
            self.send_button.setEnabled(True)
            self.load_model_button.setEnabled(True)
            self.model_selector.setEnabled(True)
            self._append_system(f"✅ {status}")
        elif "Failed" in status or "not found" in status:
            self.load_model_button.setEnabled(True)
            self.model_selector.setEnabled(True)
            self.send_button.setEnabled(False)
            self._append_system(f"❌ Error: {status}")
            QMessageBox.warning(self, "Model Load Error", status)

# -------------------------
# Main entry
# -------------------------
def main():
    app = QApplication(sys.argv)
    window = TOOChatApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()