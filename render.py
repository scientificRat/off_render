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
                    float lum = abs(dot(normalize(light), normalize(v_norm)));
                    lum = clamp(45.0/(d_light*(d_light+0.02)) * lum, 0.0,1.0)* 0.6 +0.3;
                    f_color = vec4(lum * vec3(1.0, 1.0, 1.0), 0.0);
                }
            ''',
        )

        self.vbo_vertices = None
        self.vbo_normals = None
        self.vao = None
        self.fbo = None
        # uniform variables
        self.light = self.prog['Light']
        self.mvp = self.prog['Mvp']

    def setViewport(self, viewport):
        self.ctx.viewport = viewport

    def load_model(self, vertices, normals):
        vertices = vertices.flatten()
        normals = normals.flatten()
        if self.vbo_vertices is not None:
            self.vbo_vertices.release()
        if self.vbo_normals is not None:
            self.vbo_normals.release()
        self.vbo_vertices = self.ctx.buffer(vertices.astype(np.float32).tobytes())
        self.vbo_normals = self.ctx.buffer(normals.astype(np.float32).tobytes())
        if self.vao is not None:
            self.vao.release()
        self.vao = self.ctx.vertex_array(self.prog, [
            (self.vbo_vertices, '3f', 'in_vert'),
            (self.vbo_normals, '3f', 'in_norm'),
        ])

    def render_frame(self, angle):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        camera_r = 3.88  # >= 1 / sin(pi/12)
        light_r = 6.5
        phi = 30 / 180 * np.pi
        cos_angle, sin_angle, cos_phi, sin_phi = np.cos(angle), np.sin(angle), np.cos(phi), np.sin(phi)
        camera_pos = (cos_angle * cos_phi * camera_r, sin_angle * cos_phi * camera_r, sin_phi * camera_r)
        self.light.value = (cos_angle * cos_phi * light_r, sin_angle * cos_phi * light_r, sin_phi * light_r)

        proj = Matrix44.perspective_projection(30.0, 1, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_pos,
            (0.0, 0.0, 0.0),  # look at origin
            (0.0, 0.0, 1.0),  # camera orientation
        )
        self.mvp.write((proj * lookat).astype('f4').tobytes())
        self.vao.render()

    def render_to_images(self, output_views=12):
        delta_angle = 2 * np.pi / output_views
        if self.fbo is None:
            self.fbo = self.ctx.simple_framebuffer((1024, 1024))
        self.fbo.use()
        images = []
        for i in range(output_views):
            angle = delta_angle * i
            self.render_frame(angle)
            image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
            images.append(image)
        self.fbo.clear()
        return images

    def render_and_save(self, off_file, output_dir, output_views=12):
        self.load_model(*ol.load_off(off_file))
        images = self.render_to_images(output_views)
        self._save_images(images, off_file, output_dir)

    # def _save_images_in_parallel(self, images, off_file, output_dir):
    #     import threading as th
    #     th.Thread(target=Render._save_images(images, off_file, output_dir)).start()

    @staticmethod
    def _save_images(images, off_file, output_dir):
        for i, image in enumerate(images):
            image = image.resize((299, 299), Image.BICUBIC)
            image.save("%s/%s_%03d.jpg" % (output_dir, off_file.split('.')[0].split('/')[-1], i))


def main():
    render = Render()
    off_file = "/home/scientificrat/modelnet/ModelNet40/cone/train/cone_0117.off"
    print("loading model...")
    model = ol.load_off(off_file)
    render.load_model(*model)
    print("start render...")
    images = render.render_to_images()
    for i, image in enumerate(images):
        image = image.resize((512, 512), Image.BICUBIC)
        image.save("out-%s.jpg" % i)
    print("finished")


if __name__ == '__main__':
    main()
