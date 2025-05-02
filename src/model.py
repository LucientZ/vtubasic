from rendering import *
from typing import Union
import json

class Model:
    _root_shape: Shape
    _layers: dict[str, Shape]
    _config: dict
    _model_view: MatrixStack

    def __init__(self, model_directory: str):
        self._model_view = MatrixStack()
        self.load(model_directory)

    def load(self, model_directory) -> None:
        with open(f"{model_directory}/config.json", "r") as config_file:
            self._config = json.loads(config_file.read())

        self._layers = {}
        for i, layer_info in enumerate(reversed(self._config["parts"])):
            texture = Texture(f"{model_directory}/{layer_info["texture"]}")
            with open(f"{model_directory}/{layer_info["mesh"]}", "r") as mesh_file:
                mesh_config = json.loads(mesh_file.read())
                triangle_indices = mesh_config["triangles"]
                vertices: list[Vertex] = []
                layer_compression_factor = 20.0 # Makes each layer closer together
                for vertex_info in mesh_config["vertices"]:
                    vertex = Vertex(
                        vertex_info["pos"][0],
                        vertex_info["pos"][1],
                        -float(i / len(self._config["parts"]) / layer_compression_factor),
                        vertex_info["texPos"][0],
                        vertex_info["texPos"][1]
                    )
                    vertices.append(vertex)
                shape = Shape(vertices, triangle_indices, texture, layer_info.get("name"))
                self._layers[layer_info["name"]] = shape
        
        self._root_shape = self._layers[self._config["hierarchy"]["root"]]
        for (parent_name, child_names) in self._config["hierarchy"]["relations"].items():
            for child_name in child_names:
                self._layers[parent_name].add_child_shape(self._layers[child_name])

    def draw(self, program: Program):
        self._root_shape.apply_transformation_hierarchy(self._model_view)
        for layer in self.get_layers():
            layer.draw(program)

    def get_layers(self):
        return self._layers.values()
