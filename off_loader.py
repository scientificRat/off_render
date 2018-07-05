import numpy as np
import re


def load_off(file_name):
    np.seterr(all='raise')
    f = open(file_name, 'r')
    lines = f.readlines()
    if lines[0] == 'OFF\n':
        start_line = 2
    elif re.match('^OFF', lines[0]) is not None:
        start_line = 1
    else:
        raise IOError("NOT OFF FILE")
    split_strings = [line.rstrip().split(' ') for line in lines]
    vertices = []
    normals = []
    out_vertices = []
    state = 0
    centroid = np.array([0.0, 0.0, 0.0])
    weight_sum = 0
    for i in range(start_line, len(split_strings)):
        arr = split_strings[i]
        if len(arr) == 3 and state == 0:
            vertex = [float(v) for v in arr]
            vertices.append(np.array(vertex))
        elif len(arr) == 4:
            state = 1
            c, v1, v2, v3 = arr
            assert c == '3'
            v1, v2, v3 = int(v1), int(v2), int(v3)
            l21 = vertices[v1] - vertices[v2]
            l32 = vertices[v2] - vertices[v3]
            l13 = vertices[v3] - vertices[v1]
            face_centroid = (vertices[v1] + vertices[v2] + vertices[v1]) / 3
            weight = np.linalg.norm(np.cross(l13, l21))
            weight_sum += weight
            centroid += face_centroid * weight
            out_vertices += [vertices[v1], vertices[v2], vertices[v3]]
            normals += [normalize(np.cross(l13, l21)), normalize(np.cross(l21, l32)), normalize(np.cross(l32, l13))]
        else:
            raise IOError('wrong file format')
    f.close()
    out_vertices = np.array(out_vertices)
    normals = np.array(normals)
    out_vertices -= centroid / weight_sum
    max_length = 0
    for vertex in out_vertices:
        length = np.linalg.norm(vertex, ord=2)
        if length > max_length:
            max_length = length
    out_vertices /= max_length
    return out_vertices, normals


def normalize(v):
    return v / np.linalg.norm(v)


def main():
    vertices, normals = load_off('/home/scientificrat/modelnet/ModelNet40/cone/train/cone_0117.off')
    print(vertices)
    print(normals)


if __name__ == '__main__':
    main()
