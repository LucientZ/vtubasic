import pygame, pynput
import sys, os
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
    _model: Union[Model, DebugModel, None]
    _draw_wireframe: bool = False

    # Realtime data that is useful when the user interacts with the window itself
    _normalized_mouse_pos: tuple[float, float] # normalized coordinates of mouse position from 0 to 1. Puts (0, 0) on bottom left
    _ndc_mouse_pos: tuple[float, float] # normalized device coordinates of mouse position from -1 to 1

    # Realtime data that affects model deformations
    _mouse_controller: pynput.mouse.Controller
    _desktop_ndc_mouse_pos: tuple[float, float] # normalized device coordinates including outside of the window. This is effectively the same as ndc_mouse_coordinates.
    _desktop_width: int
    _desktop_height: int
    _window_x: int
    _window_y: int

    def __init__(self):
        self._physics_thread = None


        # Initializes Window
        initial_window_pos = (100, 100)
        self._window_x, self._window_y = initial_window_pos
        os.environ["SDL_VIDEO_WINDOW_POS"] = "%d, %d" % initial_window_pos
        
        pygame.init()
        pygame.display.set_mode((self._window_width, self._window_height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self._clock = pygame.time.Clock()
        pygame.display.set_caption("VTubasic")
        self._desktop_width, self._desktop_height = pygame.display.get_desktop_sizes()[0]

        # Configure OpenGL
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)

        self.set_bg_color(0.0, 1.0, 0.0, 1.0)
        self._program = Program()
        self._program.compile_shaders("resources/vertex_shader.glsl", "resources/fragment_shader.glsl")
        self._program.bind()
        self._program.add_uniform("modelView")
        self._program.add_uniform("textureOffset")
        self._shapes = []
        self._threads = []
        self._mouse_controller = pynput.mouse.Controller()
        Shape._WIREFRAME_TEXTURE = Texture("resources/SolidRed.png")
        glLineWidth(2.0)

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
                mouse_pos = pygame.mouse.get_pos()
                self._normalized_mouse_pos = (
                    mouse_pos[0] / self._window_width,
                    (self._window_height - mouse_pos[1]) / self._window_height
                )
                self._ndc_mouse_pos = (
                     (2 * mouse_pos[0] / self._window_width) - 1,
                     -((2 * mouse_pos[1] / self._window_height) - 1)
                )
                
        
                # Calculates how many pixels from being centered
                margin_x = (self._desktop_width - self._window_width) / 2.0
                margin_y = (self._desktop_height - self._window_height) / 2.0
                offset_x = margin_x - self._window_x
                offset_y = margin_y - self._window_y
                                
                real_x, real_y = self._mouse_controller.position # Top left of the screen

                real_x += offset_x
                real_y += offset_y

                self._desktop_ndc_mouse_pos = (
                    2 * real_x / self._desktop_width - 1,
                    -(2 * real_y / self._desktop_height - 1)
                )

                self.before_render()

                # Update display
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Start drawing stuff
                for shape in self._shapes:
                    shape.draw(self._program)

                if self._model != None:
                    self._model.draw(self._program, self._draw_wireframe)

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
            elif event.type == pygame.VIDEORESIZE:
                self._window_width = event.w
                self._window_height = event.h

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
            elif event.type == pygame.VIDEORESIZE:
                self._window_width = event.w
                self._window_height = event.h
            elif event.type == pygame.WINDOWMOVED:
                self._window_x = event.x
                self._window_y = event.y
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
        delta_time = self._physics_clock.tick() / 1000

        look_position = self._desktop_ndc_mouse_pos
        for shape in self._shapes:
            shape.apply_deformers(delta_time=delta_time, mouse_pos=look_position)
        
        for shape in self._model.get_layers():
            shape.apply_deformers(delta_time=delta_time, mouse_pos=look_position)


    def quit(self) -> None:
        if self._physics_thread != None:
            self._physics_thread.join()
        super().quit()

class EditorApp(App):
    _layer_names: list[str]
    _selected_layer: int
    _ctrl_held: bool

    def __init__(self):
        super().__init__()
        self._selected_layer = 0
        self._triangulators = []
        self.set_bg_color(0.5,0.5,0.5,1.0)
        self._ctrl_held = False

    def before_render(self):
        previous_layer = self._selected_layer
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.VIDEORESIZE:
                self._window_width = event.w
                self._window_height = event.h
            elif event.type == pygame.MOUSEBUTTONDOWN:
                assert(isinstance(self._model, DebugModel))
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # left mouse down
                    self._model._current_triangulator.add_vertex(Vertex(
                        self._ndc_mouse_pos[0],
                        self._ndc_mouse_pos[1],
                        -1.0,
                        self._normalized_mouse_pos[0],
                        self._normalized_mouse_pos[1]
                    ))
                
            elif event.type == pygame.KEYDOWN:
                assert(isinstance(self._model, DebugModel))
                if event.key == pygame.K_LEFTBRACKET:
                    self._selected_layer -= 1
                elif event.key == pygame.K_RIGHTBRACKET:
                    self._selected_layer += 1
                elif event.key == pygame.K_r:
                    self._model._current_triangulator.reset()
                elif event.key == pygame.K_z:
                    self._model._current_triangulator.pop_vertex()
                elif event.key == pygame.K_s and self._ctrl_held:
                    self._model._current_triangulator.save_triangles()
                elif event.key == pygame.K_LCTRL or event.key == pygame.K_RCTRL:
                    self._ctrl_held = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LCTRL or event.key == pygame.K_RCTRL:
                    self._ctrl_held = False
        
        # Changes selected layer
        if self._model != None and previous_layer != self._selected_layer:
            self._selected_layer = self._selected_layer % len(self._layer_names)
            self._model.change_root(self._layer_names[self._selected_layer])

    def load_model(self, directory_path: str):
        model = DebugModel(directory_path)
        self._model = model
        self._layer_names = list(map(lambda x: x.get_name(), model.get_layers()))
        self._model.change_root(self._layer_names[self._selected_layer])