from rendering import *
import json
from OpenGL.GL import *

class Model:
    _root_shape: Shape
    _layers: dict[str, Shape]
    _config: dict
    _model_view: MatrixStack
    _model_directory: str

    def __init__(self, model_directory: str):
        self._model_view = MatrixStack()
        self._model_directory = model_directory
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
                if layer_info.get("deformers") != None:
                    for deformer_file in layer_info.get("deformers"):
                        with open(f"{model_directory}/{deformer_file}", "r") as f:
                            deformer_config = json.loads(f.read())
                            if deformer_config["type"] == "cloth":
                                shape.add_dynamic_deformer(ClothDeformer(
                                    shape,
                                    deformer_config["dynamicVertices"],
                                    [
                                        numpy.array([0.0, -deformer_config["gravity"], 0.0]),
                                        numpy.array([0.0, 0.0, 0.0]) # Used for intertial forces
                                    ],
                                    deformer_config["damping"],
                                    deformer_config["alpha"],
                                    deformer_config["mass"]
                                ))
                            elif deformer_config["type"] == "position":
                                shape.add_static_deformer(PositionDeformer(
                                    shape,
                                    (deformer_config["xLowerBound"], deformer_config["xUpperBound"]),
                                    (deformer_config["yLowerBound"], deformer_config["yUpperBound"])
                                ))

        
        self._root_shape = self._layers[self._config["hierarchy"]["root"]]
        for (parent_name, child_names) in self._config["hierarchy"]["relations"].items():
            for child_name in child_names:
                self._layers[parent_name].add_child_shape(self._layers[child_name])

    def draw(self, program: Program, draw_wireframe: bool = False):
        self._root_shape.apply_transformation_hierarchy(self._model_view)
        for layer in self.get_layers():
            layer.draw(program, draw_wireframe)

    def draw_layer(self, name: str, program: Program):
        """
        Draws a specific layer.
        This can be handy when viewing one specific piece of a model
        """
        self._layers[name].draw(program)

    def change_root(self, name: str):
        """
        Changes the root shape of the model
        """
        self._root_shape = self._layers[name]
        print(self._root_shape)

    def get_layers(self):
        return self._layers.values()

    def get_model_directory(self) -> str:
        return self._model_directory

    def get_config(self) -> dict:
        return self._config
    
    def reset(self):
        for layer in self.get_layers():
            layer.reset()
    
class DebugModel(Model):
    """
    Model which is more useful for debugging and editing.
    Loads all textures as squares and only draws root shape.
    """
    def __init__(self, model_directory: str):
        super().__init__(model_directory)
    
    def draw(self, program: Program, draw_wireframe: bool = False):
        self._root_shape.draw(program, draw_wireframe)

    def load(self, model_directory) -> None:
        with open(f"{model_directory}/config.json", "r") as config_file:
            self._config = json.loads(config_file.read())

        self._layers = {}
        for layer_info in reversed(self._config["parts"]):
            texture = Texture(f"{model_directory}/{layer_info["texture"]}")
            shape = ImageShape(texture, layer_info.get("name"))
            self._layers[layer_info["name"]] = shape
        
        self._root_shape = self._layers[self._config["hierarchy"]["root"]]
        for (parent_name, child_names) in self._config["hierarchy"]["relations"].items():
            for child_name in child_names:
                self._layers[parent_name].add_child_shape(self._layers[child_name])