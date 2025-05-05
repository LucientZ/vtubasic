import sys, os, json
sys.path.insert(0, "./src")
from app import *
from prompt import *
from model import DEFAULT_MODEL_CONFIGURATION, DEFAULT_MESH, DEFAULT_POSITION_DEFORMER
import shutil

def create_model():
    model_name = input("Enter a name for the model: ")
    model_path = f"models/{model_name}"

    if len(model_name) == 0:
        print("Model name is empty", file=sys.stderr)
        exit(1)

    if os.path.exists(model_path):
        print("This model name already exists", file=sys.stderr)
        exit(1)

    os.makedirs(model_path)
    os.makedirs(f"{model_path}/deformers")
    os.makedirs(f"{model_path}/textures")
    os.makedirs(f"{model_path}/meshes")

    with open(f"{model_path}/config.json", "w") as f:
        json.dump(DEFAULT_MODEL_CONFIGURATION, f)
    
    with open(f"{model_path}/meshes/bodyMesh.json", "w") as f:
        json.dump(DEFAULT_MESH, f)

    with open(f"{model_path}/deformers/bodyPosition.json", "w") as f:
        json.dump(DEFAULT_POSITION_DEFORMER, f)

    shutil.copy("resources/DefaultGuy.png", f"{model_path}/textures/Body.png")

    exit(0)

def create_model_path_factory(path: str):
    """
    Workaround for getting filepath because for some reason lambdas didn't work
    """
    def path_factory():
        return path
    return path_factory

if __name__ == "__main__":
    model_prompts = [
        ("Create a new model from scratch", create_model)
    ]
    for (i, directory) in enumerate(os.listdir("models")):
        model_prompts.append(
            (directory, create_model_path_factory(f"models/{directory}")) # For some reason lambdas kept using the last result, so this is a workaround
        )

    model_directory = ask_prompt(model_prompts, "Choose a model to use:")
    app_prompts = [
        ("Start runtime app", lambda: RuntimeApp()),
        ("Start editor app (Edit vertices)", lambda: EditorApp()),
        ("Quit", exit)
    ]
    print(f"Loading {model_directory}")
    app: App = ask_prompt(app_prompts)
    app.load_model(model_directory)
    app.run()