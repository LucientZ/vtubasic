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
        for layer_info in reversed(self._config["parts"]):
            texture = Texture(f"{model_directory}/{layer_info["texture"]}")
            with open(f"{model_directory}/{layer_info["mesh"]}", "r") as mesh_file:
                mesh_config = json.loads(mesh_file.read())
                triangle_indices = mesh_config["triangles"]
                vertices: list[Vertex] = []
                for vertex_info in mesh_config["vertices"]:
                    vertex = Vertex(
                        vertex_info["vertex"][0],
                        vertex_info["vertex"][1],
                        vertex_info["vertex"][2],
                        vertex_info["vertex"][3],
                        vertex_info["vertex"][4]
                    )
                    vertices.append(vertex)
                base_shape = Shape(vertices, triangle_indices, texture)
                self._layers.append(base_shape)
    
    def get_layers(self):
        return self._layers
