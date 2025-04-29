import pygame
import sys
from OpenGL.GL import *


class App:
    _window_width: int = 640
    _window_height: int = 480
    _framerate: int = 60
    _clock: pygame.time.Clock
    _running: bool = False

    def __init__(self):
        # Initializes Window
        pygame.init()
        pygame.display.set_mode((self._window_width, self._window_height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self._clock = pygame.time.Clock()
        glClearColor(0.1, 0.2, 0.2, 1)

    def run(self):
        self._running = True
        self.renderLoop()


    def renderLoop(self):
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
            pygame.display.flip()

            self._clock.tick(self._framerate)

        self.quit()

    def quit(self):
        pygame.quit()

                    
