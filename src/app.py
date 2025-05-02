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
    _framerate: int = 30
    _clock: pygame.time.Clock
    _running: bool = False
    _program: Program
    _shapes: list[Shape]
    _threads: list[Thread]
    _model: Model
    _normalized_mouse_pos: tuple[float, float] # normalized device coordinates of mouse position

    def __init__(self):
        self._physics_thread = None
        # Initializes Window
        pygame.init()
        pygame.display.set_mode((self._window_width, self._window_height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self._clock = pygame.time.Clock()
        self.set_bg_color(0.0, 1.0, 0.0, 1.0)
        pygame.display.set_caption("VTubasic")
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._program = Program()
        self._program.compile_shaders("resources/vertex_shader.glsl", "resources/fragment_shader.glsl")
        self._program.bind()
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
                glClear(GL_COLOR_BUFFER_BIT)

                # Start drawing stuff
                self._program.bind()
                for shape in self._shapes:
                    shape.draw()
                self._program.unbind()

                pygame.display.flip()
                self._clock.tick(self._framerate)
        except KeyboardInterrupt:
            self._running = False

        self.quit()

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
        self._shapes = self._model.get_layers()
        self._running = True
        physics_thread = Thread(target=self.physics_loop)
        self._threads.append(physics_thread)
        physics_thread.start()
        super().run()

    def physics_loop(self) -> None:
        while(self._running):
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