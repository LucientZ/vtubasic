import pygame
import numpy
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram
from collections import namedtuple
from typing import Union, Any
import json

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

def vertex_to_numpy_array(vertex: Vertex):
    return numpy.array([vertex.x, vertex.y, vertex.z])

class Texture:
    _texture: Any
    def __init__(self, filepath: str):
        self.load(filepath)

    def load(self, filepath: str) -> None:
        image = pygame.transform.flip(pygame.image.load(filepath).convert_alpha(), False, True)
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

class Deformer():
    def __init__(self):
        pass
    def apply(self, h = 0.0) -> None:
        """
        Applies the deformer's rules to the original shape
        """
        pass
    
class Shape:
    _vertices: list[Vertex] # Structured information about the shape
    _original_vertices: list[Vertex] # Copy of vertices if a reset is required
    _triangle_indices: list[int]
    _vertex_buf: numpy.ndarray[tuple[int], numpy.dtype[numpy.floating]] # Flattened version of vertices
    _index_buf: numpy.ndarray[tuple[int], numpy.dtype[numpy.integer]]
    _vertex_count: int
    _vao: Any
    _vbo: Any # Vertex buffer object
    _ebo: Any # Element buffer object
    _texture: Union[Texture, None]
    _static_deformers: list[Deformer] # Deformers which temporarily modify vertices. These send information to the GPU
    _dynamic_deformers: list[Deformer] # Deformers which permanently modify vertices. These mutate the given shape

    def __init__(self, vertices: list[Vertex], triangle_indices: list[int], texture: Texture = None):
        """
        Creates a Shape object with given parameters
        
        Parameters:
            - vertices: list of vertex information including position and texture coordinates
            - triangle_indices: index values defining each triangle
            - texture: Texture object defining what texture tha shape samples from
        """
        self._vertices = vertices.copy()
        self._original_vertices = vertices.copy()
        self._triangle_indices = triangle_indices
        self._index_buf = numpy.array(triangle_indices, dtype=numpy.uint32)
        self._texture = texture
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        self._static_deformers = []
        self._dynamic_deformers = []

        self._ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._index_buf.nbytes, self._index_buf, GL_STATIC_DRAW)

    def draw(self):
        if self._texture != None:
            self._texture.bind()

        # Create and bind buffer data
        self.create_vertex_buffers()
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, self._vertex_buf.nbytes, self._vertex_buf, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_SIZE_BYTES, ctypes.c_void_p(POSITION_OFFSET))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_SIZE_BYTES, ctypes.c_void_p(TEXTURE_OFFSET))

        self.apply_static_deformers()

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glDrawElements(GL_TRIANGLES, len(self._triangle_indices), GL_UNSIGNED_INT, ctypes.c_void_p(0))

        if self._texture != None:
            self._texture.unbind()

    def create_vertex_buffers(self) -> None:
        self._vertex_count = len(self._vertices)
        self._vertex_buf = numpy.array(self._vertices, dtype=numpy.float32).flatten()

    def apply_static_deformers(self):
        """
        Sends static deformer information to the GPU.
        """
        for deformer in self._static_deformers:
            deformer.apply()

    def apply_dynamic_deformers(self, h = 0.1):
        """
        Mutates the current shape with its dynamic deformers.
        This is used for physics calculations.
        """
        for deformer in self._dynamic_deformers:
            deformer.apply(h)

    def add_static_deformer(self, deformer: Deformer):
        self._static_deformers.append(deformer)

    def add_dynamic_deformer(self, deformer: Deformer):
        self._dynamic_deformers.append(deformer)

    def get_triangle_indices(self) -> list[int]:
        return self._triangle_indices
    
    def get_vertex(self, index: int) -> Vertex:
        return self._vertices[index]

    def set_vertex(self, index: int, vertex: Union[Vertex, numpy.ndarray]) -> None:
        if isinstance(vertex, Vertex):
            self._vertices[index] = vertex
        else:
            original_vertex = self._vertices[index]
            self._vertices[index] = Vertex(
                vertex[0],
                vertex[1],
                vertex[2],
                original_vertex.s,
                original_vertex.t,
            )

    def get_all_vertices(self) -> list[Vertex]:
        return self._vertices

    def reset(self) -> None:
        self._vertices = self._original_vertices.copy()

    def destroy(self):
        """
        Removes all buffers
        """
        glDeleteVertexArrays(1, (self._vao,))
        glDeleteBuffers(1, (self._vbo,))
        glDeleteBuffers(1, (self._ebo,))

class Particle:
    damping: float
    x0: numpy.ndarray # Initial position
    v0: numpy.ndarray # Initial velocity
    x: numpy.ndarray # Current Position
    p: numpy.ndarray # Previous position
    v: numpy.ndarray # Current velocity
    fixed: bool

    def __init__(self, damping: float, position: Vertex, fixed: bool = True):
        self.damping = damping
        self.x0 = vertex_to_numpy_array(position)
        self.x = vertex_to_numpy_array(position)
        self.p = vertex_to_numpy_array(position)
        self.v0 = numpy.array([0.0,0.0,0.0])
        self.v = numpy.array([0.0,0.0,0.0])
        self.fixed = fixed

class SpringConstraint:
    alpha: float
    length: float
    p0: Particle
    p1: Particle

    def __init__(self, p0: Particle, p1: Particle, alpha: float):
        self.p0 = p0
        self.p1 = p1
        self.alpha = alpha
        self.length = numpy.linalg.norm(p0.x0 - p1.x0)
        pass

class ClothDeformer(Deformer):
    """
    Used for simulating things like hair or clothing
    This deformer mutates the given shape
    """
    _dynamic_vertex_indices: list[int]
    _forces: numpy.ndarray # List of 2d forces applied to all dynamic vertices
    _shape: Shape # reference to shape to modify
    _springs: list[SpringConstraint]
    _particles: list[Particle]

    def __init__(self,
                shape: Shape,
                dynamic_indices: list[int] = None, 
                forces: numpy.ndarray = None,
                damping: float = 1e-5,
                alpha: float = 1e-5):
        self._shape = shape
        self._dynamic_vertex_indices = dynamic_indices if dynamic_indices != None else []
        self._forces = forces if forces != None else []
        self._alpha = alpha
        self._springs = []
        self._particles = []

        for i, vertex in enumerate(shape.get_all_vertices()):
            self._particles.append(Particle(damping, vertex, not (i in dynamic_indices)))

        relations: dict[int, set[int]] = {}
        triangle_indices = shape.get_triangle_indices()
        # Figure out what springs need to be created
        for i in range(0, len(triangle_indices), 3):
            v1 = triangle_indices[i]
            v2 = triangle_indices[i+1]
            v3 = triangle_indices[i+2]
            
            if relations.get(min(v1, v2)) == None:
                relations[min(v1, v2)] = {max(v1, v2)}
            else:
                if not max(v1, v2) in relations.get(min(v1, v2)):
                    relations.get(min(v1, v2)).add(max(v1, v2))
            
            if relations.get(min(v1, v3)) == None:
                relations[min(v1, v3)] = {max(v1, v3)}
            else:
                if not max(v1, v3) in relations.get(min(v1, v3)):
                    relations.get(min(v1, v3)).add(max(v1, v3))
            
            if relations.get(min(v3, v2)) == None:
                relations[min(v3, v2)] = {max(v3, v2)}
            else:
                if not max(v3, v2) in relations.get(min(v3, v2)):
                    relations.get(min(v3, v2)).add(max(v3, v2))

        for key, value in relations.items():
            p0 = self._particles[key]
            for index in value:
                p1 = self._particles[index]
                self._springs.append(SpringConstraint(p0, p1, alpha))

    def apply(self, h = 0.1):
        # Apply forces to particles
        for index in self._dynamic_vertex_indices:
            particle = self._particles[index]
            for force in self._forces:
                fi = force - particle.damping * particle.v
                particle.v += h * fi
                particle.p = particle.x.copy()
                particle.x = particle.x + h * particle.v

        # Apply spring constraints
        for spring in self._springs:
            w0 = 1.0
            w1 = 1.0
            delta_x = spring.p1.x - spring.p0.x
            l = numpy.linalg.norm(delta_x)
            C = l - spring.length
            gradC0 = -delta_x / l
            gradC1 = delta_x / l

            lambda_value = -C / (w0 + w1 + spring.alpha / (h * h))

            if not spring.p0.fixed:
                spring.p0.x += lambda_value * w0 * gradC0
            if not spring.p1.fixed:
                spring.p1.x += lambda_value * w1 * gradC1

        # Update velocities
        for index in self._dynamic_vertex_indices:
            particle = self._particles[index]
            particle.v = 1.0/h * (particle.x - particle.p)
        
        # Applies calculated positions
        for index in self._dynamic_vertex_indices:
            particle = self._particles[index]
            self._shape.set_vertex(index, particle.x)

class ImageShape(Shape):
    """
    Image texture that spans the entire screen
    """
    def __init__(self, texture: Union[Texture, None] = None):
        vertices = [
            Vertex(-1.0, -1.0, 0.0, 0.0, 0.0),
            Vertex(1.0, -1.0, 0.0, 1.0, 0.0),
            Vertex(-1.0, 1.0, 0.0, 0.0, 1.0),
            Vertex(1.0, 1.0, 0.0, 1.0, 1.0),
        ]

        indices = [
            0, 1, 3,
            0, 3, 2
        ]
        super().__init__(vertices, indices, texture)
