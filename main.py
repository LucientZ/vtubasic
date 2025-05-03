import sys, os
sys.path.insert(0, "./src")
from app import *
from prompt import *

if __name__ == "__main__":
    model_prompts = [
        (directory, lambda: f"models/{directory}")
        for directory in os.listdir("models")
    ]
    model_directory = ask_prompt(model_prompts, "Choose a model to use:")
    app_prompts = [
        ("Start runtime app", lambda: RuntimeApp()),
        ("Start editor app (Broken)", lambda: EditorApp()),
    ]
    app: App = ask_prompt(app_prompts)
    app.load_model(model_directory)
    app.run()