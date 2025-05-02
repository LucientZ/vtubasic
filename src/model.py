from rendering import *
import json

class Model:
    _layers: list[Shape]
    _config: dict
    def __init__(self, model_directory: str):
        self.load(model_directory)

    def load(self, model_directory) -> None:
        with open(f"{model_directory}/config.json", "r") as config_file:
            self._config = json.loads(config_file.read())

        self._layers = []
        for i, layer_info in enumerate(reversed(self._config["parts"])):
            texture = Texture(f"{model_directory}/{layer_info["texture"]}")
            self._layers.append(ImageShape(texture))
    
    def get_layers(self):
        return self._layers
