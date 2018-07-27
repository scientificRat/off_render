import time

import numpy as np
import off_loader as ol
import moderngl
from PyQt5 import QtOpenGL, QtWidgets
import render
import argparse


class WindowInfo:
    def __init__(self):
        self.size = (0, 0)
        self.mouse = (0, 0)
        self.wheel = 0
        self.time = 0
        self.ratio = 1.0
        self.viewport = (0, 0, 0, 0)
        self.keys = np.full(256, False)
        self.old_keys = np.copy(self.keys)

    def key_down(self, key):
        return self.keys[key]

    def key_pressed(self, key):
        return self.keys[key] and not self.old_keys[key]

    def key_released(self, key):
        return not self.keys[key] and self.old_keys[key]


class RenderWindow(QtOpenGL.QGLWidget):
    def __init__(self, render_class, off_file, size=(512, 512), title="off_render"):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSwapInterval(1)
        fmt.setSampleBuffers(True)
        fmt.setDepthBufferSize(24)

        super(RenderWindow, self).__init__(fmt, None)
        self.setFixedSize(size[0], size[1])
        self.move(QtWidgets.QDesktopWidget().rect().center() - self.rect().center())
        self.setWindowTitle(title)

        self.model = ol.load_off(off_file)

        self.start_time = time.time()
        self.render_class = render_class
        self.render = None

        self.wnd = WindowInfo()
        self.wnd.viewport = (0, 0) + size
        self.wnd.ratio = size[0] / size[1]
        self.wnd.size = size

    def keyPressEvent(self, event):
        self.wnd.keys[event.nativeVirtualKey() & 0xFF] = True

    def keyReleaseEvent(self, event):
        self.wnd.keys[event.nativeVirtualKey() & 0xFF] = False

    def mouseMoveEvent(self, event):
        self.wnd.mouse = (event.x(), event.y())

    def wheelEvent(self, event):
        self.wnd.wheel += event.angleDelta().y()

    def paintGL(self):
        if self.render is None:
            ctx = moderngl.create_context()
            self.render = self.render_class(ctx=ctx)
            self.render.load_model(*self.model)
            self.render.setViewport(self.wnd.viewport)
        self.wnd.time = time.time() - self.start_time
        angle = 0.9 * self.wnd.time
        self.render.render_frame(angle)
        self.wnd.old_keys = np.copy(self.wnd.keys)
        self.wnd.wheel = 0
        self.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='OFF_FILE', help='the off_file you want to render')
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    render_window = RenderWindow(render.Render, args.file)
    render_window.show()
    app.exec_()
    del app
