import pygame
import numpy
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram
from collections import namedtuple
from typing import Union

class Program:
    _shader: ShaderProgram

    def __init__(self):
        pass
    
    def compile_shaders(self, vertex_filepath: str, fragment_filepath: str) -> None:
        vertex_src: str
        fragment_src: str
        with open(vertex_filepath, "r") as f:
            vertex_src = f.readlines()
        
        with open(fragment_filepath, "r") as f:
            fragment_src = f.readlines()

        self._shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )

    def bind(self):
        glUseProgram(self._shader)

    def unbind(self):
        glUseProgram(0)

    def destroy(self):
        glDeleteProgram(self._shader)


Vertex = namedtuple("Vertex", "x y z s t")
VERTEX_SIZE_BYTES = 5 * 4
POSITION_OFFSET = 0
TEXTURE_OFFSET = 12

class Texture:
    _texture: any
    def __init__(self, filepath: str):
        self.load(filepath)

    def load(self, filepath: str) -> None:
        image = pygame.image.load(filepath)
        image_width, image_height = image.get_rect().size
        image_data = pygame.image.tostring(image, "RGBA")

        self._texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

        self.unbind()

    def bind(self) -> None:
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._texture)

    def unbind(self) -> None:
        glBindTexture(GL_TEXTURE_2D, 0) # Unbind texture

    def destroy(self) -> None:
        glDeleteTextures(1, (self._texture))

class Shape:
    _vertices: list[Vertex] # Structured information about the shape
    _vertex_buf: numpy.ndarray[tuple[int], numpy.dtype[numpy.floating]] # Flattened version of vertices
    _vertex_count: int
    _vao: any
    _vbo: any # Vertex buffer object
    _texture: Union[Texture, None] = None

    def __init__(self, vertices: list[Vertex], texture: Union[Texture, None] = None):
        self._vertices = vertices
        self._texture = texture

        self.create_vertex_buf()
        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)
        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, self._vertex_buf.nbytes, self._vertex_buf, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_SIZE_BYTES, ctypes.c_void_p(POSITION_OFFSET))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_SIZE_BYTES, ctypes.c_void_p(TEXTURE_OFFSET))

    def draw(self):
        if self._texture != None:
            self._texture.bind()

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)

        if self._texture != None:
            self._texture.unbind()

    def create_vertex_buf(self) -> None:
        self._vertex_count = len(self._vertices)
        self._vertex_buf = numpy.array(self._vertices, dtype=numpy.float32).flatten()

    def destroy(self):
        glDeleteVertexArrays(1, (self._vao,))
        glDeleteBuffers(1, (self._vbo,))


class TestTriangle(Shape): 
    def __init__(self, texture: Union[Texture, None] = None):
        vertex1: Vertex = Vertex(-0.5, -0.5, 0.0, 0.0, 1.0)
        vertex2: Vertex = Vertex(0.5,  -0.5, 0.0, 1.0, 1.0)
        vertex3: Vertex = Vertex(0.0,  0.5,  0.0, 0.5, 0.0)
        super().__init__([vertex1, vertex2, vertex3], texture)


class Deformer:
    _shape: Shape
    _pos_buf: list[float] = []
    _tex_buf: list[float] = []

    def __init__(self, shape: Shape):
        self._shape = shape
        pass