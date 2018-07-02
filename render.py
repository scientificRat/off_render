import off_loader as ol
import moderngl
import numpy as np
from pyrr import Matrix44
from PIL import Image


class Render(object):
    def __init__(self, ctx=None):
        if ctx is None:
            self.ctx = moderngl.create_standalone_context()
        else:
            self.ctx = ctx
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_vert;
                in vec3 in_norm;
                out vec3 v_vert;
                out vec3 v_norm;
                void main() {
                    v_vert =  in_vert;
                    v_norm =  in_norm;
                    gl_Position = Mvp*vec4(v_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 Light;
                in vec3 v_vert;
                in vec3 v_norm;
                out vec4 f_color;
                void main() {
                    vec3 light = Light - v_vert;
                    float d_light = length(light);
                    float lum = clamp(abs(dot(normalize(light), normalize(v_norm))), 0.0, 1.0) * 0.6 + 0.3;
                    lum = clamp(60.0/(d_light*(d_light+0.03)) * lum, 0.0,1.0);
                    f_color = vec4(lum * vec3(1.0, 1.0, 1.0), 0.0);
                }
            ''',
        )
        self.vao = None
        self.light = None
        self.mvp = None

    def setViewport(self, viewport):
        self.ctx.viewport = viewport

    def load_model(self, vertices, normals):
        vertices = vertices.flatten()
        normals = normals.flatten()
        vbo_vertices = self.ctx.buffer(vertices.astype(np.float32).tobytes())
        vbo_normals = self.ctx.buffer(normals.astype(np.float32).tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (vbo_vertices, '3f', 'in_vert'),
            (vbo_normals, '3f', 'in_norm'),
        ])
        # uniform variables
        self.light = self.prog['Light']
        self.mvp = self.prog['Mvp']
        pass

    def render_frame(self, angle):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        camera_pos = (np.cos(angle) * 3, np.sin(angle) * 3, np.sin(30 / 180 * np.pi) * 3)
        # light.value = (0, 0, 0.5)
        self.light.value = (2.3 * camera_pos[0], 2.3 * camera_pos[1], 2.3 * camera_pos[2])

        proj = Matrix44.perspective_projection(45.0, 1, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_pos,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        self.mvp.write((proj * lookat).astype('f4').tobytes())
        self.vao.render()

    def render_to_image(self, output_views=12):
        delta_angle = 2 * np.pi / output_views
        fbo = self.ctx.simple_framebuffer((2048, 2048))
        fbo.use()
        for i in range(output_views):
            angle = delta_angle * i
            self.render_frame(angle)
            image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
            image.resize((512, 512))
            image.save("output/out-%s.jpg" % i)


def main():
    render = Render()
    off_file = "/Volumes/EXTEND_SD/ModelNet10/bed/train/bed_0063.off"
    print("loading model...")
    model = ol.load_off(off_file)
    render.load_model(*model)
    print("start render...")
    render.render_to_image()
    print("finished")


if __name__ == '__main__':
    main()
