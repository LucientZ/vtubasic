import pygame
import numpy
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram
from collections import namedtuple
from typing import Union, Any
from math import sin, cos, sqrt
import json

class Program:
    _shader: ShaderProgram
    _uniform_variables: dict[str, Any]

    def __init__(self):
        self._uniform_variables = {}
    
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

    def add_uniform(self, name: str):
        self._uniform_variables[name] = glGetUniformLocation(self._shader, name)

    def get_uniform(self, name: str) -> Any:
        return self._uniform_variables[name]

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

class MatrixStack:
    _stack: list[numpy.matrix]

    def __init__(self):
        top_matrix = numpy.identity(3, dtype=numpy.float32)
        self._stack = [top_matrix]

    def get_top_matrix(self):
        return self._stack[-1]

    def translate(self, x: float, y: float):
        translation_matrix = numpy.matrix(
            [[1.0, 0.0, x],
             [0.0, 1.0, y],
             [0.0, 0.0, 1.0]],
            dtype=numpy.float32
        )
        self._stack[-1] = translation_matrix * self._stack[-1]

    def rotate(self, theta_x = 0.0, theta_y = 0.0, theta_z = 0.0):
        """
        Rotates the matrix in radians
        """
        x_rotation_matrix = numpy.matrix(
            [[1.0, 0.0,          0.0],
             [0.0, cos(theta_x), -sin(theta_x)],
             [0.0, sin(theta_x), cos(theta_x)]],
            dtype=numpy.float32
        )
        y_rotation_matrix = numpy.matrix(
            [[cos(theta_y),  0.0, sin(theta_y)],
             [0.0,           1.0, 0.0],
             [-sin(theta_y), 0.0, cos(theta_y)]],
            dtype=numpy.float32
        )
        z_rotation_matrix = numpy.matrix(
            [[cos(theta_z),  -sin(theta_z), 0.0],
             [sin(theta_z),   cos(theta_z), 0.0],
             [0.0,            0.0,          1.0]],
            dtype=numpy.float32
        )
        self._stack[-1] = x_rotation_matrix * y_rotation_matrix * z_rotation_matrix * self._stack[-1]

    def push_matrix(self) -> None:
        self._stack.append(self._stack[-1])

    def pop_matrix(self) -> numpy.matrix:
        if(len(self._stack) <= 1):
            raise Exception("Matrix stack cannot pop anymore matrices")
        return self._stack.pop()

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
    """
    Interface for deformers which change the shape of a model
    """
    def apply(self, *args, **kwargs) -> None:
        """
        Applies the deformer's rules to the original shape
        """
        pass

    def reset(self):
        pass
    
class Shape:
    _vertices: list[Vertex] # Structured information about the shape
    _original_vertices: list[Vertex] # Copy of vertices if a reset is required
    _triangle_indices: list[int]
    _vertex_buf: numpy.ndarray[tuple[int], numpy.dtype[numpy.floating]] # Flattened version of vertices
    _index_buf: numpy.ndarray[tuple[int], numpy.dtype[numpy.integer]]
    _vbo: Any # Vertex buffer object
    _ebo: Any # Element buffer object
    _texture: Union[Texture, None]
    _deformers: list[Deformer] # Deformers which change the shape
    _child_shapes: list["Shape"] # Nested shapes to apply transformations to
    _name: str
    _translation: list[float]
    _transformation_matrix: numpy.matrix
    _texture_offset: numpy.ndarray
    _WIREFRAME_TEXTURE: Texture # Static variable used when drawing wireframe. Initialized when pygame is finished

    def __init__(self, vertices: list[Vertex], triangle_indices: list[int], texture: Texture = None, name: str = None, texture_offset: list[float] = None):
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
        self._texture = texture if texture != None else Shape._WIREFRAME_TEXTURE
        self._vbo = glGenBuffers(1)
        self._ebo = glGenBuffers(1)
        self._deformers = []
        self._child_shapes = []
        self._translation = [0.0, 0.0]
        self._rotation = [0.0, 0.0, 0.0]
        self._texture_offset = numpy.array([texture_offset[0], texture_offset[1]], dtype=numpy.float32) if texture_offset != None else numpy.array([0.0, 0.0], dtype=numpy.float32) 
        self._transformation_matrix = numpy.identity(3, dtype=numpy.float32)
        self._name = name if name != None else "Shape"

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._index_buf.nbytes, self._index_buf, GL_STATIC_DRAW)

    def draw(self, program: Program, draw_wireframe: bool = False, redraw_shape: bool = True):
        program.bind()
        if self._texture != None:
            Shape._WIREFRAME_TEXTURE.bind() if draw_wireframe else self._texture.bind()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if draw_wireframe else GL_FILL)

        # Create and bind buffer data to GPU
        self.create_vertex_buffers()
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, self._vertex_buf.nbytes, self._vertex_buf, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_SIZE_BYTES, ctypes.c_void_p(POSITION_OFFSET))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, VERTEX_SIZE_BYTES, ctypes.c_void_p(TEXTURE_OFFSET))

        # Send transformation data to GPU
        glUniformMatrix3fv(
            program.get_uniform("modelView"),
            1,
            GL_TRUE,
            self._transformation_matrix
        )
        glUniform2fv(
            program.get_uniform("textureOffset"),
            1,
            self._texture_offset
        )

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glDrawElements(GL_TRIANGLES, len(self._triangle_indices), GL_UNSIGNED_INT, ctypes.c_void_p(0))

        # Draw shape normally after wireframe
        if draw_wireframe and redraw_shape:
            self.draw(program, False)

        if self._texture != None:
            self._texture.unbind()
        program.unbind()

    def apply_transformation_hierarchy(self, model_view: MatrixStack) -> None:
        model_view.push_matrix()
        model_view.rotate(self._rotation[0], self._rotation[1], self._rotation[2])
        model_view.translate(self._translation[0], self._translation[1])
        previous_matrix = self._transformation_matrix
        self._transformation_matrix = model_view.get_top_matrix()
        for shape in self._child_shapes:
            shape.apply_transformation_hierarchy(model_view)

        # Apply potential forces to each deformer that needs it
        for deformer in self._deformers:
            if isinstance(deformer, ClothDeformer):
                delta_transform: numpy.matrix = self._transformation_matrix - previous_matrix
                scaling_factor = 300 # Used to make forces stronger
                force = numpy.array([
                    scaling_factor * (delta_transform.item((0, 0)) + delta_transform.item((0, 1)) + delta_transform.item((0, 2))),
                    scaling_factor * (delta_transform.item((1, 0)) + delta_transform.item((1, 1)) + delta_transform.item((1, 2))),
                    0.0
                ])
                deformer.set_force(1, force)
        model_view.pop_matrix()

    def create_vertex_buffers(self) -> None:
        self._vertex_buf = numpy.array(self._vertices, dtype=numpy.float32).flatten()

    def apply_deformers(self, **kwargs) -> None:
        self._translation = [0.0, 0.0, 0.0]
        self._rotation = [0.0, 0.0, 0.0]
        self._texture_offset = numpy.array([0.0, 0.0], dtype=numpy.float32)
        for deformer in self._deformers:
            deformer.apply(**kwargs)

    def add_deformer(self, deformer: Deformer) -> None:
        self._deformers.append(deformer)

    def add_child_shape(self, shape: "Shape") -> None:
        self._child_shapes.append(shape)

    def set_program(self, program: Program) -> None:
        self._program = program

    def get_triangle_indices(self) -> list[int]:
        return self._triangle_indices
    
    def get_vertex(self, index: int) -> Vertex:
        return self._vertices[index]
    
    def translate(self, translation: tuple[float, float]) -> None:
        self._translation[0] += translation[0]
        self._translation[1] += translation[1]

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

    def set_texture_offset(self, offset: numpy.ndarray) -> None:
        self._texture_offset = offset

    def get_texture_offset(self) -> numpy.ndarray:
        return self._texture_offset

    def get_all_vertices(self) -> list[Vertex]:
        return self._vertices
    
    def get_name(self) -> str:
        return self._name

    def reset(self) -> None:
        for deformer in self._deformers:
            deformer.reset()
        self._vertices = self._original_vertices.copy()

    def destroy(self):
        """
        Removes all buffers
        """
        glDeleteBuffers(1, (self._vbo,))
        glDeleteBuffers(1, (self._ebo,))

    def __str__(self):
        return self._name

class Particle:
    damping: float
    mass: float
    x0: numpy.ndarray # Initial position
    v0: numpy.ndarray # Initial velocity
    x: numpy.ndarray # Current Position
    p: numpy.ndarray # Previous position
    v: numpy.ndarray # Current velocity
    fixed: bool

    def __init__(self, damping: float, position: Vertex, mass: float, fixed: bool = True):
        self.damping = damping
        self.x0 = vertex_to_numpy_array(position)
        self.x = vertex_to_numpy_array(position)
        self.p = vertex_to_numpy_array(position)
        self.v0 = numpy.array([0.0,0.0,0.0])
        self.v = numpy.array([0.0,0.0,0.0])
        self.fixed = fixed
        self.mass = mass

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
    _forces: list[numpy.ndarray] # List of 2d forces applied to all dynamic vertices
    _shape: Shape # reference to shape to modify
    _springs: list[SpringConstraint]
    _particles: list[Particle]
    _alpha: float
    _damping: float
    _time_modifier: float # Changes how quickly or slowly physics happens

    def __init__(self,
                shape: Shape,
                dynamic_indices: list[int] = None, 
                forces: list[numpy.ndarray] = None,
                damping: float = 1e-5,
                alpha: float = 1e-5,
                mass: float = 10.0,
                time_modifier = 1.0):
        self._shape = shape
        self._dynamic_vertex_indices = dynamic_indices if dynamic_indices != None else []
        self._forces = forces if forces != None else []
        self._alpha = alpha
        self._damping = damping
        self._time_modifier = time_modifier
        self._springs = []
        self._particles = []

        for i, vertex in enumerate(shape.get_all_vertices()):
            self._particles.append(Particle(damping, vertex, mass, not (i in dynamic_indices)))

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

    def apply(self, delta_time: float = 0.1, **kwargs):
        if delta_time > 0.1: # This usually means there was a lag spike
            return
        delta_time = delta_time * self._time_modifier
        # Apply forces to particles
        for index in self._dynamic_vertex_indices:
            particle = self._particles[index]
            force_sum = sum(self._forces)
            fi = force_sum - particle.damping * particle.v
            particle.v += delta_time * fi
            particle.p = particle.x.copy()
            particle.x = particle.x + delta_time * particle.v

        # Apply spring constraints
        for spring in self._springs:
            w0 = 1.0 / spring.p0.mass
            w1 = 1.0 / spring.p1.mass
            delta_x = spring.p1.x - spring.p0.x
            l = numpy.linalg.norm(delta_x)
            C = l - spring.length
            gradC0 = -delta_x / l
            gradC1 = delta_x / l

            lambda_value = -C / (w0 + w1 + spring.alpha / (delta_time * delta_time))

            if not spring.p0.fixed:
                spring.p0.x += lambda_value * w0 * gradC0
            if not spring.p1.fixed:
                spring.p1.x += lambda_value * w1 * gradC1

        # Update velocities
        for index in self._dynamic_vertex_indices:
            particle = self._particles[index]
            particle.v = 1.0/delta_time * (particle.x - particle.p)
        
        # Applies calculated positions
        for index in self._dynamic_vertex_indices:
            particle = self._particles[index]
            self._shape.set_vertex(index, particle.x)

    def set_force(self, index: int, force: numpy.ndarray):
        self._forces[index] = force

    def reset(self):
        for particle in self._particles:
            particle.x = particle.x0.copy()
            particle.p = particle.x0.copy()
            particle.v = particle.v0.copy()
        
class PositionDeformer(Deformer):
    _x_bounds: tuple[float, float]
    _y_bounds: tuple[float, float]
    _x_min: Union[float, None]
    _x_max: Union[float, None]
    _y_min: Union[float, None]
    _y_max: Union[float, None]
    _shape: Shape
    _bind: str

    def __init__(self,
                shape: Shape,
                x_bounds: tuple[float, float],
                y_bounds: tuple[float, float],
                x_min: Union[float, None] = None,
                x_max: Union[float, None] = None,
                y_min: Union[float, None] = None,
                y_max: Union[float, None] = None,
                bind: str = None):
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._shape = shape
        self._bind = bind

    def apply(self, mouse_pos: tuple[float, float] = (0, 0), **_):
        x_value: float = 0.0
        y_value: float = 0.0
        if self._bind == "mouse":
            # Convert normalized device coordinates with a range [-1.0,1.0] to specified range
            x_value = (mouse_pos[0] + 1.0) / 2.0 * (self._x_bounds[1] - self._x_bounds[0]) + self._x_bounds[0]
            y_value = (mouse_pos[1] + 1.0) / 2.0 * (self._y_bounds[1] - self._y_bounds[0]) + self._y_bounds[0]
        elif self._bind == "time":
            print("hi") # TODO Fix this

        if self._x_min != None:
            x_value = max(x_value, self._x_min)
        
        if self._x_max != None:
            x_value = min(x_value, self._x_max)

        if self._y_min != None:
            y_value = max(y_value, self._y_min)
        
        if self._y_max != None:
            y_value = min(y_value, self._y_max)

        self._shape.translate((x_value, y_value))

class TextureDeformer(Deformer):
    """
    Class which replaces a texture depending on what expression the program is currently using.
    Note: this class *adds* onto current texture meaning the offset of a shape must be reset before a render cycle
    """
    _shape: Shape
    _expression: str
    _texture_offset: numpy.ndarray
    def __init__(self, shape: Shape, texture_offset: numpy.ndarray, expression: str):
        self._shape = shape
        self._texture_offset = texture_offset
        self._expression = expression

    def apply(self, expression: str, **_):
        if expression == self._expression:
            old_offset = self._shape.get_texture_offset()
            self._shape.set_texture_offset(self._texture_offset + old_offset) 

class TextureAnimationDeformer(Deformer):
    """
    Class which creates texture-coordinate based animations.
    Note: this class *adds* onto current texture meaning the offset of a shape must be reset before a render cycle
    """

    _shape: Shape
    _keyframes: list[numpy.ndarray] # List of different offsets
    _timing: list[float]
    _duration_seconds: float
    _time: float

    def __init__(self, shape: Shape, keyframes: list[numpy.ndarray], timing: list[float], duration_seconds: float):
        assert(len(keyframes) == len(timing))
        self._shape = shape
        self._keyframes = keyframes
        self._timing = timing
        self._duration_seconds = duration_seconds
        self._time = 0.0

    def apply(self, delta_time: float, **_):
        self._time += delta_time
        timing_offset = self._time % self._duration_seconds # Get which point in the animation we are at

        # Find the keyframe that should play in this time
        for (i) in range(len(self._timing)):
            prev_time = self._timing[(i-1) % len(self._keyframes)]
            next_time = self._timing[(i+1) % len(self._keyframes)]
            if (timing_offset > prev_time or i == 0) and (timing_offset < next_time or i == (len(self._keyframes) - 1)):
                texture_offset = self._keyframes[i]
                old_offset = self._shape.get_texture_offset()
                self._shape.set_texture_offset(texture_offset + old_offset) 
                break

class ImageShape(Shape):
    """
    Image texture that spans the entire screen
    """
    def __init__(self, texture: Union[Texture, None] = None, name: str = None):
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
        super().__init__(vertices, indices, texture, name)


class AutoTriangulator():
    MAX_VERTEX_DISTANCE: float = 0.22

    class Edge():
        v0: Vertex
        v1: Vertex
        v0_index: int
        v1_index: int
        def __init__(self, v0: Vertex, v1: Vertex, v0_index: int, v1_index: int):
            self.v0 = v0
            self.v1 = v1
            self.v0_index = v0_index
            self.v1_index = v1_index
        
        def __eq__(self, value):
            return (self.v0_index == value.v0_index and self.v1_index == value.v1_index) or (self.v0_index == value.v1_index and self.v1_index == value.v0_index)
        
        def __hash__(self):
            return hash(frozenset((self.v0_index, self.v1_index)))

    class Triangle():
        v0: Vertex
        v1: Vertex
        v2: Vertex
        v0_index: int
        v1_index: int
        v2_index: int
        a: float
        b: float
        c: float
        radius: float
        circumcenter: Vertex

        def __init__(self, v0: Vertex, v1: Vertex, v2: Vertex, v0_index: int, v1_index: int, v2_index: int):

            self.v0 = v0
            self.v1 = v1
            self.v2 = v2
            self.v0_index = v0_index
            self.v1_index = v1_index
            self.v2_index = v2_index

            A = self.v0
            B = self.v1
            C = self.v2

            # Calculate sidelengths of triangle
            a = sqrt((A.x-B.x)**2 + (A.y-B.y)**2)
            b = sqrt((A.x-C.x)**2 + (A.y-C.y)**2)
            c = sqrt((C.x-B.x)**2 + (C.y-B.y)**2)

            self.a = a
            self.b = b
            self.c = c
            self.radius = a*b*c / sqrt((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))
            
            # https://en.wikipedia.org/wiki/Circumcircle#Circumcenter_coordinates
            D = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))
            self.circumcenter = Vertex(
                1/D * ((A.x**2 + A.y**2)*(B.y - C.y) + (B.x**2 + B.y**2)*(C.y - A.y) + (C.x**2 + C.y**2)*(A.y - B.y)),
                1/D * ((A.x**2 + A.y**2)*(C.x - B.x) + (B.x**2 + B.y**2)*(A.x - C.x) + (C.x**2 + C.y**2)*(B.x - A.x)),
                0.0,
                0.0,
                1.0
            )

        def is_in_circumcircle(self, vertex: Vertex) -> bool:
            dx = self.circumcenter.x - vertex.x
            dy = self.circumcenter.y - vertex.y
            return sqrt(dx*dx + dy*dy) <= self.radius

    """
    Abstracts the concepts of creating triangles from arbitrary vertices
    """
    _vertices: list[Vertex]
    _triangle_indices: list[int]
    _original_vertices: list[Vertex]
    _original_triangle_indices: list[int]
    _shape: Shape
    _mesh_file: str

    def __init__(self, vertices: list[Vertex], triangle_indices: list[int], mesh_file: str):
        self._vertices = vertices.copy()
        self._triangle_indices = triangle_indices.copy()
        self._original_vertices = vertices.copy()
        self._original_triangle_indices = triangle_indices.copy()
        self._mesh_file = mesh_file
        self._shape = Shape(self._vertices, self._triangle_indices)

    def auto_triangulate(self):
        """
        Implementation of Delaunay Triangulation which also has a certain distance at which triangles can connect
        https://www.gorillasun.de/blog/bowyer-watson-algorithm-for-delaunay-triangulation/

        TODO Fix issue where for some reason this tries to connect every vertex
        """
        if len(self._vertices) < 3:
            print(f"Mesh has {len(self._vertices)} vertices")
            self._triangle_indices = []
            self._shape._triangle_indices = []
            self._shape._vertices = self._vertices
            return
        # Create a super triangle that encapsulates all vertices which will be removed later
        
        super_triangle_0 = Vertex(-10.0, -10.0, 0.0, 0.0, 0.0)
        super_triangle_1 = Vertex( 10.0, -10.0, 0.0, 0.0, 0.0)
        super_triangle_2 = Vertex( 0.0,   10.0, 0.0, 0.0, 0.0)
        
        triangles: list[AutoTriangulator.Triangle] = [AutoTriangulator.Triangle(
            super_triangle_0,
            super_triangle_1,
            super_triangle_2,
            -1,
            -2,
            -3
        )]

        for (i, vertex) in enumerate(self._vertices):
            good_triangles: list[AutoTriangulator.Triangle] = []
            edges: list[AutoTriangulator.Edge] = []
            for triangle in triangles:
                if triangle.is_in_circumcircle(vertex):
                    edges.append(AutoTriangulator.Edge(
                        triangle.v0,
                        triangle.v1,
                        triangle.v0_index,
                        triangle.v1_index
                    ))
                    edges.append(AutoTriangulator.Edge(
                        triangle.v0,
                        triangle.v2,
                        triangle.v0_index,
                        triangle.v2_index
                    ))
                    edges.append(AutoTriangulator.Edge(
                        triangle.v2,
                        triangle.v1,
                        triangle.v2_index,
                        triangle.v1_index
                    ))
                else:
                    good_triangles.append(triangle)

            edges = list(set(edges)) # Gets unique edges

            for edge in edges:
                good_triangles.append(AutoTriangulator.Triangle(
                    edge.v0,
                    edge.v1,
                    vertex,
                    edge.v0_index,
                    edge.v1_index,
                    i
                ))
            
            triangles = good_triangles

        triangle_indices: list[int] = []
        for triangle in triangles:
            # Skip triangles involving super triangle
            if triangle.v0_index < 0 or triangle.v1_index < 0 or triangle.v2_index < 0:
                continue

            # Skip triangles that are too large
            if triangle.a > AutoTriangulator.MAX_VERTEX_DISTANCE or triangle.b > AutoTriangulator.MAX_VERTEX_DISTANCE or triangle.c > AutoTriangulator.MAX_VERTEX_DISTANCE:
                continue

            triangle_indices.append(triangle.v0_index)
            triangle_indices.append(triangle.v1_index)
            triangle_indices.append(triangle.v2_index)
        self._triangle_indices = triangle_indices
        self._shape = Shape(self._vertices, self._triangle_indices)

    def reset(self):
        print("resetting")
        self._vertices = self._original_vertices.copy()
        self._triangle_indices = self._original_triangle_indices.copy()
        self._shape = Shape(self._vertices, self._triangle_indices)

    def add_vertex(self, vertex: Vertex):
        self._vertices.append(vertex)
        print(f"Vertex number {len(self._vertices) - 1}")
        try:
            self.auto_triangulate()
        except Exception as e:
            print(f"Issue adding vertex: {e}")
            self._vertices.pop()
            

    def pop_vertex(self):
        if len(self._vertices) > 0:
            self._vertices.pop()
        self.auto_triangulate()

    def save_triangles(self):
        """
        Overwrites currently loaded mesh
        """
        print(f"Saved to {self._mesh_file}")
        with open(self._mesh_file, "w") as f:
            json.dump({
                "triangles": self._triangle_indices,
                "vertices": list(map(lambda v: {
                    "pos": [v.x, v.y],
                    "texPos": [v.s, v.t]
                }, self._vertices))
            }, f)

    def draw(self, program: Program):
        self._shape.draw(program, draw_wireframe=True, redraw_shape=False)

