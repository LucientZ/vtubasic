import pygame
import sys
from rendering import *
from prompt import *
from model import *
from OpenGL.GL import *
from threading import Thread

class App:
    _window_width: int = 1280
    _window_height: int = 720
    _framerate: int = 60
    _clock: pygame.time.Clock
    _running: bool = False
    _program: Program
    _shapes: list[Shape]
    _threads: list[Thread]
    _model: Union[Model, None]
    _normalized_mouse_pos: tuple[float, float] # normalized coordinates of mouse position from 0 to 1. Puts (0, 0) on bottom left
    _ndc_mouse_pos: tuple[float, float] # normalized device coordinates of mouse position from -1 to 1
    _draw_wireframe: bool = False

    def __init__(self):
        self._physics_thread = None

        # Initializes Window
        pygame.init()
        pygame.display.set_mode((self._window_width, self._window_height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self._clock = pygame.time.Clock()
        pygame.display.set_caption("VTubasic")

        # Configure OpenGL
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)

        self.set_bg_color(0.0, 1.0, 0.0, 1.0)
        self._program = Program()
        self._program.compile_shaders("resources/vertex_shader.glsl", "resources/fragment_shader.glsl")
        self._program.bind()
        self._program.add_uniform("modelView")
        self._shapes = []
        self._threads = []
        Shape._WIREFRAME_TEXTURE = Texture("resources/SolidRed.png")
        glLineWidth(3.0)

    def run(self) -> None:
        self._running = True
        self.render_loop()
        
    def set_bg_color(self, r: float, g: float, b: float, a: float) -> None:
        glClearColor(r, g, b, a)

    def render_loop(self) -> None:
        """
        Renders the application
        """
        if not self._running:
            print("Render loop called while program not running", file=sys.stderr)
            return
        try:
            while(self._running):
                # Set user input values
                window_info = pygame.display.Info()
                self._window_height = window_info.current_h
                self._window_width = window_info.current_w
                mouse_pos = pygame.mouse.get_pos()
                self._normalized_mouse_pos = (
                    mouse_pos[0] / self._window_width,
                    (self._window_height - mouse_pos[1]) / self._window_height
                )
                self._ndc_mouse_pos = (
                     (2 * mouse_pos[0] / self._window_width) - 1,
                     -((2 * mouse_pos[1] / self._window_height) - 1)
                )

                self.before_render()

                # Update display
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Start drawing stuff
                for shape in self._shapes:
                    shape.draw(self._program)

                if self._model != None:
                    self._model.draw(self._program, self._draw_wireframe)
                self._program.unbind()

                self.after_render()
                pygame.display.flip()
                self._clock.tick(self._framerate)
        except KeyboardInterrupt:
            self._running = False

        self.quit()

    def before_render(self):
        """
        Override this to put stuff before each render.
        By default this simply tests if pygame has been exited
        """
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                self._running = False

    def after_render(self):
        """
        Override this to put stuff after each render
        """
        pass

    def load_model(self, directory_path: str):
        model = Model(directory_path)
        self._model = model
    
    def quit(self):
        for shape in self._shapes:
            shape.destroy()
        for thread in self._threads:
            thread.join()
        self._program.destroy()
        pygame.quit()

class RuntimeApp(App):
    _physics_clock: pygame.time.Clock
    _follow_mouse: bool

    def __init__(self):
        self._follow_mouse = False
        self._physics_clock = pygame.time.Clock()
        super().__init__()

    def run(self) -> None:
        self._running = True
        super().run()

    def before_render(self):
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                self._running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # left mouse down
                self._follow_mouse = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: # left mouse up
                self._follow_mouse = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self._draw_wireframe = not self._draw_wireframe
                if event.key == pygame.K_r:
                    if self._model != None:
                        self._model.reset()

    def after_render(self):
        seconds = self._physics_clock.tick() / 1000
        for shape in self._shapes:
            shape.apply_static_deformers(mouse_pos=self._ndc_mouse_pos, follow_mouse=self._follow_mouse)
            shape.apply_dynamic_deformers(seconds)
        
        for shape in self._model.get_layers():
            shape.apply_static_deformers(mouse_pos=self._ndc_mouse_pos, follow_mouse=self._follow_mouse)
            shape.apply_dynamic_deformers(seconds)


    def quit(self) -> None:
        if self._physics_thread != None:
            self._physics_thread.join()
        super().quit()

class EditorApp(App):
    _triangulators: list[AutoTriangulator]
    _layer_names: list[str]
    _selected_layer: int
    _ui_program: Program

    def __init__(self):
        super().__init__()
        self._selected_layer = 0
        self._triangulators = []
        self._ui_program = Program()
        self._ui_program.compile_shaders("resources/wireframe_vertex_shader.glsl", "resources/wireframe_fragment_shader.glsl")
        self._ui_program.add_uniform("lineColor")
        self.set_bg_color(0.5,0.5,0.5,1.0)

    def before_render(self):
        previous_layer = self._selected_layer
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                self._running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print("\"pos\":", self._ndc_mouse_pos)
                print("\"texPos\":", self._normalized_mouse_pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFTBRACKET:
                    self._selected_layer -= 1
                elif event.key == pygame.K_RIGHTBRACKET:
                    self._selected_layer += 1
                elif event.key == pygame.K_w:
                    self._draw_wireframe = not self._draw_wireframe
        if self._model != None and previous_layer != self._selected_layer:
            self._selected_layer = self._selected_layer % len(self._layer_names)
            self._model.change_root(self._layer_names[self._selected_layer])

        self._triangulators[self._selected_layer].draw(self._ui_program)

    def after_render(self):
        self._triangulators[self._selected_layer].draw(self._ui_program)

    def load_model(self, directory_path: str):
        model = DebugModel(directory_path)
        self._model = model
        self._layer_names = list(map(lambda x: x.get_name(), model.get_layers()))
        self._model.change_root(self._layer_names[self._selected_layer])
        self._ui_program.bind()
        for shape in model.get_layers():
            self._triangulators.append(AutoTriangulator(shape.get_all_vertices(), shape.get_triangle_indices()))
        self._ui_program.unbind()