import pygame
import sys, time
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
    _normalized_mouse_pos: tuple[float, float] # normalized device coordinates of mouse position

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
                self.before_render()
                for event in pygame.event.get():
                    if(event.type == pygame.QUIT):
                        self._running = False

                # Set user input values
                window_info = pygame.display.Info()
                self._window_height = window_info.current_h
                self._window_width = window_info.current_w
                mouse_pos = pygame.mouse.get_pos()
                self._normalized_mouse_pos = (
                    mouse_pos[0] / self._window_width,
                    mouse_pos[1] / self._window_height
                )

                # Update display
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Start drawing stuff
                self._program.bind()
                for shape in self._shapes:
                    shape.draw(self._program)

                if self._model != None:
                    self._model.draw(self._program)
                self._program.unbind()

                pygame.display.flip()
                self._clock.tick(self._framerate)
                self.after_render()
        except KeyboardInterrupt:
            self._running = False

        self.quit()

    def before_render(self):
        """
        Override this to put stuff before each render
        """
        pass

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

    def __init__(self):
        self._physics_clock = pygame.time.Clock()
        super().__init__()

    def run(self) -> None:
        self._running = True
        super().run()

    def after_render(self):
        self.do_physics()

    def do_physics(self) -> None:
        for shape in self._shapes:
            seconds = self._physics_clock.tick(self._framerate) / 1000
            shape.apply_dynamic_deformers(seconds)

    def quit(self) -> None:
        if self._physics_thread != None:
            self._physics_thread.join()
        super().quit()

class EditorApp(App):
    _layers: list[Shape]
    def __init__(self):
        super().__init__()

    def run(self):
        super().run()