import off_loader as ol
import moderngl
import numpy as np
from pyrr import Matrix44
from PIL import Image

OUTPUT_VIEWS = 12
OFF_FILE = 'data/bed_0027.off'

ctx = moderngl.create_standalone_context()

prog = ctx.program(
    vertex_shader='''
        #version 330
        uniform mat4 Mvp;
        in vec3 in_vert;
        in vec3 in_norm;
        out vec3 v_vert;
        out vec3 v_norm;
        out vec3 v_color;
        void main() {
            v_vert =  in_vert/10;
            v_norm =  in_norm;
            v_color = vec3(1.0,1.0,1.0);
            gl_Position = Mvp*vec4(v_vert, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330
        uniform vec3 Light;
        in vec3 v_vert;
        in vec3 v_norm;
        in vec3 v_color;
        out vec4 f_color;
        void main() {
            float lum = clamp(abs(dot(normalize(Light - v_vert), normalize(v_norm))), 0.0, 1.0) * 0.8 + 0.2;
            f_color = vec4(v_color*lum, 1.0);
        }
    ''',
)

vertices, normals = ol.load_off(OFF_FILE)
vertices = vertices.flatten()
normals = normals.flatten()
print(len(vertices), len(normals))
light = prog['Light']
mvp = prog['Mvp']
vbo_vertices = ctx.buffer(vertices.astype(np.float32).tobytes())
vbo_normals = ctx.buffer(normals.astype(np.float32).tobytes())
vao = ctx.vertex_array(prog, [
    (vbo_vertices, '3f', 'in_vert'),
    (vbo_normals, '3f', 'in_norm'),
])

delta_angle = 2 * np.pi / OUTPUT_VIEWS

for i in range(OUTPUT_VIEWS):
    angle = delta_angle * i

    fbo = ctx.simple_framebuffer((2048, 2048))
    fbo.use()

    ctx.clear(1.0, 1.0, 1.0)
    ctx.enable(moderngl.DEPTH_TEST)

    camera_pos = (np.cos(angle) * 20.0, np.sin(angle) * 20.0, 5.0)
    light.value = camera_pos

    proj = Matrix44.perspective_projection(45.0, 1, 0.1, 1000.0)
    lookat = Matrix44.look_at(
        camera_pos,
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 1.0),
    )

    mvp.write((proj * lookat).astype('f4').tobytes())
    vao.render()
    image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
    image.save("output/out-%s.jpg" % i)
