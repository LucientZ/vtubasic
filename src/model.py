from rendering import *
import json, os
from OpenGL.GL import *

class Model:
    _root_shape: Shape
    _layers: dict[str, Shape]
    _config: dict
    _model_view: MatrixStack
    _model_directory: str
    _textures: dict[str, Texture]
    _expressions: list[str]

    def __init__(self, model_directory: str):
        self._model_view = MatrixStack()
        self._model_directory = model_directory
        self.load(model_directory)

    def load(self, model_directory) -> None:
        with open(f"{model_directory}/config.json", "r") as config_file:
            self._config = json.loads(config_file.read())

        self._textures = {}
        self._layers = {}
        self._expressions = self._config.get("expressions") if self._config.get("expressions") != None else []

        for i, layer_info in enumerate(reversed(self._config["parts"])):
            existing_texture = self._textures.get(layer_info["mesh"])
            texture = existing_texture if existing_texture != None else Texture(f"{model_directory}/{layer_info["texture"]}") 
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
                shape = Shape(vertices, triangle_indices, texture, layer_info.get("name"), layer_info.get("textureOffset"))
                self._layers[layer_info["name"]] = shape
                if layer_info.get("deformers") != None:
                    for deformer_file in layer_info.get("deformers"):
                        with open(f"{model_directory}/{deformer_file}", "r") as f:
                            deformer_config = json.loads(f.read())
                            if deformer_config["type"] == "cloth":
                                shape.add_deformer(ClothDeformer(
                                    shape,
                                    deformer_config["dynamicVertices"],
                                    [
                                        numpy.array([0.0, -deformer_config["gravity"], 0.0]),
                                        numpy.array([0.0, 0.0, 0.0]) # Used for intertial forces
                                    ],
                                    deformer_config["damping"],
                                    deformer_config["alpha"],
                                    mass=deformer_config["mass"],
                                    time_modifier=deformer_config.get("timeModifier") if deformer_config.get("timeModifier") != None else 1.0
                                ))
                            elif deformer_config["type"] == "position":
                                shape.add_deformer(PositionDeformer(
                                    shape,
                                    (deformer_config["xLowerBound"], deformer_config["xUpperBound"]),
                                    (deformer_config["yLowerBound"], deformer_config["yUpperBound"]),
                                    x_min = deformer_config.get("xMin"),
                                    x_max = deformer_config.get("xMax"),
                                    y_min = deformer_config.get("yMin"),
                                    y_max = deformer_config.get("yMax"),
                                    bind = deformer_config.get("bind")
                                ))
                            elif deformer_config["type"] == "textureAnimation":
                                shape.add_deformer(TextureAnimationDeformer(
                                    shape,
                                    list(map(lambda x: numpy.array(x, dtype=numpy.float32), deformer_config["keyframes"])),
                                    deformer_config["timing"],
                                    deformer_config["durationSeconds"]
                                ))

                if layer_info.get("expressionTextureOffsets") != None:
                    for (expression, offset) in layer_info.get("expressionTextureOffsets").items():
                        shape.add_deformer(TextureDeformer(
                            shape,
                            numpy.array(offset, dtype=numpy.float32),
                            expression
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

    def get_layers(self):
        return self._layers.values()

    def get_model_directory(self) -> str:
        return self._model_directory

    def get_config(self) -> dict:
        return self._config

    def get_expressions(self) -> list[str]:
        return self._expressions
    
    def reset(self):
        for layer in self.get_layers():
            layer.reset()
    
class DebugModel(Model):
    """
    Model which is more useful for debugging and editing.
    Loads all textures as squares and only draws root shape.
    """
    _triangulators: dict[str, AutoTriangulator]
    _current_triangulator: AutoTriangulator

    def __init__(self, model_directory: str):
        super().__init__(model_directory)

    def load(self, model_directory) -> None:
        self._triangulators = {}
        with open(f"{model_directory}/config.json", "r") as config_file:
            self._config = json.loads(config_file.read())

        self._layers = {}
        for layer_info in reversed(self._config["parts"]):
            texture = Texture(f"{model_directory}/{layer_info["texture"]}")
            shape = ImageShape(texture, layer_info.get("name"))
            self._layers[layer_info["name"]] = shape

            with open(f"{model_directory}/{layer_info["mesh"]}", "r") as mesh_file:
                    mesh_config = json.loads(mesh_file.read())
                    triangle_indices = mesh_config["triangles"]
                    vertices: list[Vertex] = []
                    for vertex_info in mesh_config["vertices"]:
                        vertex = Vertex(
                            vertex_info["pos"][0],
                            vertex_info["pos"][1],
                            -1.0,
                            vertex_info["texPos"][0],
                            vertex_info["texPos"][1]
                        )
                        vertices.append(vertex)
                    triangulator = AutoTriangulator(vertices, triangle_indices, f"{model_directory}/{layer_info["mesh"]}")
                    self._triangulators[layer_info["name"]] = triangulator
        
        self._root_shape = self._layers[self._config["hierarchy"]["root"]]
        for (parent_name, child_names) in self._config["hierarchy"]["relations"].items():
            for child_name in child_names:
                self._layers[parent_name].add_child_shape(self._layers[child_name])
        self._current_triangulator = self._triangulators[self._root_shape.get_name()]

    def draw(self, program: Program, draw_wireframe: bool = False):
        self._current_triangulator.draw(program)
        self._root_shape.draw(program, draw_wireframe=draw_wireframe)

    def change_root(self, name: str):
        print(f"{name}")
        super().change_root(name)
        self._current_triangulator = self._triangulators[name]


DEFAULT_MODEL_CONFIGURATION = {
    "name": "Default Configuration",
    "expressions": [],
    "parts": [
        {
            "name": "Body",
            "texture": "textures/Body.png",
            "mesh": "meshes/bodyMesh.json",
            "deformers": ["deformers/bodyPosition.json"]
        }
    ],
    "hierarchy": {
        "root": "Body",
        "relations": {
            "Body": []
        }
    }
}

DEFAULT_MESH = {
  "triangles": [
    0,
    1,
    3,
    0,
    3,
    2
  ],
  "vertices": [
    {
      "pos": [
        -1.0,
        -1.0
      ],
      "texPos": [
        0.0,
        0.0
      ]
    },
    {
      "pos": [
        1.0,
        -1.0
      ],
      "texPos": [
        1.0,
        0.0
      ]
    },
    {
      "pos": [
        -1.0,
        1.0
      ],
      "texPos": [
        0.0,
        1.0
      ]
    },
    {
      "pos": [
        1.0,
        1.0
      ],
      "texPos": [
        1.0,
        1.0
      ]
    },
    {
      "pos": [
        0.5,
        0.0
      ],
      "texPos": [
        0.0,
        0.0
      ]
    }
  ]
}

DEFAULT_POSITION_DEFORMER = {
    "type": "position",
    "bind": "mouse",
    "xLowerBound": -0.01,
    "xUpperBound": 0.01,
    "yLowerBound": -0.04,
    "yUpperBound": 0.0,
    "yMax": 0.0
}