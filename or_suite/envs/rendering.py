import numpy as np
import pyglet

__author__ = 'Nicolas Dickreuter'
"""
This code is adapted from Nicolas Dickreuter's render file for an OpenAI gym environment
https://github.com/dickreuter/neuron_poker/blob/master/gym_env/rendering.py
"""


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class PygletWindow:
    """Rendering class"""

    def __init__(self, X, Y):
        """Initialization"""
        self.active = True
        self.display_surface = pyglet.window.Window(width=X, height=Y + 50)
        self.top = Y

        # make OpenGL context current
        self.display_surface.switch_to()
        self.reset()

    def circle(self, x_pos, y_pos, radius, color):
        """Draw a circle"""
        y_pos = self.top - y_pos
        circle = pyglet.shapes.Circle(x_pos, y_pos, radius, color=color)
        circle.draw()

    def text(self, text, x, y, font_size=20):
        """Draw text"""
        y = self.top - y
        label = pyglet.text.Label(text, font_size=font_size,
                                  x=x, y=y, anchor_x='left', anchor_y='top')
        label.draw()

    def line(self, x1, x2, y_pos, width, color):
        y_pos = self.top - y_pos
        line = pyglet.shapes.Line(x1, y_pos, x2, y_pos, width, color)
        line.draw()

    def reset(self):
        """New frame"""
        pyglet.clock.tick()
        self.display_surface.dispatch_events()
        from pyglet.gl import glClear
        glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)

    def update(self):
        """Draw the current state on screen"""
        self.display_surface.flip()

    def close(self):
        self.display_surface.close()


if __name__ == '__main__':
    pg = PygletWindow(400, 400)

    pg.reset()
    pg.circle(5, 5, 100, 1, 5)
    pg.text("Test", 10, 10)
    pg.text("Test2", 30, 30)
    pg.update()
    input()

    pg.circle(5, 5, 100, 1, 5)
    pg.text("Test3333", 10, 10)
    pg.text("Test2123123", 303, 30)
    pg.update()
    input()