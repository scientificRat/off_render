import numpy as np


def load_off(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    if lines[0] != 'OFF\n':
        raise IOError("NOT OFF FILE")
    split_strings = [line.rstrip().split(' ') for line in lines]
    vertices = []
    normals = []
    out_vertices = []
    state = 0
    max_length = 0
    for i in range(2, len(split_strings)):
        arr = split_strings[i]
        if len(arr) == 3 and state == 0:
            vertex = [float(v) for v in arr]
            length = np.sqrt(vertex[0] ** 2 + vertex[1] ** 2 + vertex[2] ** 2)
            if length > max_length:
                max_length = length
            vertices.append(np.array(vertex))
        elif len(arr) == 4:
            state = 1
            c, v1, v2, v3 = arr
            assert c == '3'
            v1, v2, v3 = int(v1), int(v2), int(v3)
            l21 = vertices[v1] - vertices[v2]
            l32 = vertices[v2] - vertices[v3]
            l13 = vertices[v3] - vertices[v1]
            out_vertices.append(np.array([vertices[v1], vertices[v2], vertices[v3]]))
            normals.append(np.array([np.cross(l13, l21), np.cross(l21, l32), np.cross(l32, l13)]))
        else:
            raise IOError('wrong file format')
    out_vertices /= max_length
    return np.array(out_vertices), np.array(normals)


def main():
    vertices, normals = load_off('/Volumes/EXTEND_SD/ModelNet10/bed/test/bed_0516.off')
    print(vertices)
    print(normals)


if __name__ == '__main__':
    main()
