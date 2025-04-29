import pygame
import sys
from rendering import *
from OpenGL.GL import *


class App:
    _window_width: int = 640
    _window_height: int = 480
    _framerate: int = 60
    _clock: pygame.time.Clock
    _running: bool = False
    _program: Program
    _shapes: list[Shape] = []

    def __init__(self):
        # Initializes Window
        pygame.init()
        pygame.display.set_mode((self._window_width, self._window_height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self._clock = pygame.time.Clock()
        self.set_bg_color(0.0, 1.0, 0.0, 1.0)
        pygame.display.set_caption("VTubasic")

        self._program = Program()
        self._program.compile_shaders("resources/vertex_shader.glsl", "resources/fragment_shader.glsl")
        self._program.bind()
        self._shapes.append(TestTriangle())

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

        self.quit()

    def quit(self):
        for shape in self._shapes:
            shape.destroy()
        self._program.destroy()
        pygame.quit()
