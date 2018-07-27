import numpy as np
import re


def load_off(file_name):
    try:
        f = open(file_name, 'r')
    except Exception as e:
        print('load off file failed!')
        return None
    lines = f.readlines()
    if lines[0] == 'OFF\n':
        start_line = 2
    elif re.match('^OFF', lines[0]) is not None:
        start_line = 1
    else:
        raise IOError("NOT OFF FILE")
    split_strings = [line.rstrip().split(' ') for line in lines]
    vertices = []
    out_vertices = []
    state = 0
    for i in range(start_line, len(split_strings)):
        arr = split_strings[i]
        if len(arr) == 3 and state == 0:
            vertex = [float(v) for v in arr]
            vertex = np.array(vertex)
            vertices.append(vertex)
        elif len(arr) == 4:
            state = 1
            c, v1, v2, v3 = arr
            assert c == '3'
            v1, v2, v3 = int(v1), int(v2), int(v3)
            out_vertices.append([vertices[v1], vertices[v2], vertices[v3]])
        else:
            raise IOError('wrong file format')
    f.close()
    out_vertices = np.array(out_vertices)
    # to avoid overflow
    out_vertices /= np.max(np.abs(out_vertices))
    l10 = out_vertices[:, 0, :] - out_vertices[:, 1, :]
    l02 = out_vertices[:, 2, :] - out_vertices[:, 0, :]
    normals = np.cross(l10, l02)
    centroids = out_vertices.mean(1)
    weights = np.expand_dims(np.linalg.norm(normals, axis=1), 0)
    centroid = weights.dot(centroids) / weights.sum()
    out_vertices = out_vertices.reshape(-1, 3)
    out_vertices -= centroid
    max_length = np.max(np.linalg.norm(out_vertices.reshape(-1, 3), axis=1))
    out_vertices /= max_length
    return out_vertices, np.expand_dims(normals, 1).repeat(3, axis=1).reshape(-1, 3)


def main():
    vertices, normals = load_off('/home/scientificrat/modelnet/ModelNet40/cone/train/cone_0117.off')
    print(vertices)
    print(normals)


if __name__ == '__main__':
    main()
