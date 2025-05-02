import pygame
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
    
    def quit(self):
        for shape in self._shapes:
            shape.destroy()
        self._program.destroy()
        pygame.quit()

class RuntimeApp(App):
    _physics_thread: Union[Thread, None]
    
    def __init__(self):
        super().__init__()
        prompts = [
            (directory, lambda: f"models/{directory}")
            for directory in os.listdir("models")
        ]
        model_config_filename = ask_prompt(prompts, "Choose a model to use:")
        model = Model(model_config_filename)
        self._shapes = model.get_layers()

    def run(self) -> None:
        self._running = True
        self._physics_thread = Thread(target=self.physics_loop)
        self._physics_thread.start()
        super().run()

    def physics_loop(self) -> None:
        while(self._running):
            for shape in self._shapes:
                shape.apply_dynamic_deformers()
        

    def quit(self) -> None:
        if self._physics_thread != None:
            self._physics_thread.join()
        super().quit()

class EditorApp(App):
    pass