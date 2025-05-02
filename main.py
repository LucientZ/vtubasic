import sys, os
sys.path.insert(0, "./src")
from app import *
from prompt import *

if __name__ == "__main__":
    app_prompts = [
        ("Start runtime app", lambda: RuntimeApp()),
        ("Start editor app", lambda: EditorApp()),
    ]
    app: App = ask_prompt(app_prompts)
    app.run()