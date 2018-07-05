from render import Render
import os
import off_loader as ol
from PIL import Image
import multiprocessing as mp

MODEL_NET_ROOT = '/home/scientificrat/modelnet/ModelNet40'
OUTPUT_DIR = '/home/scientificrat/modelnet/o_Modelnet40'


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_off_file_in_dir(a_dir):
    for name in os.listdir(a_dir):
        arr = name.split('.')
        if len(arr) == 2 and arr[1] == 'off':
            yield name


def make_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def render_and_save(off_file, output_dir):
    render = Render()
    render.load_model(*ol.load_off(off_file))
    images = render.render_to_images()
    for i, image in enumerate(images):
        image = image.resize((299, 299), Image.BICUBIC)
        image.save("%s/%s_%03d.jpg" % (output_dir, off_file.split('.')[0].split('/')[-1], i))


def main():
    # traversal the directors
    sub_dirs = get_immediate_subdirectories(MODEL_NET_ROOT)
    process_pool = []
    for sub_dir in sub_dirs:
        # make output dirs
        out_sub_dir = OUTPUT_DIR + "/" + sub_dir
        make_dir_not_exist(out_sub_dir + "/test")
        make_dir_not_exist(out_sub_dir + "/train")
        # ready to convert
        source_dir = MODEL_NET_ROOT + "/" + sub_dir
        for d in ["/test", "/train"]:
            curr_dir = source_dir + d
            for off_file in get_off_file_in_dir(curr_dir):
                p = mp.Process(target=render_and_save, args=(curr_dir + "/" + off_file, out_sub_dir + d))
                process_pool.append(p)
    i = 0
    # simple parallel solution, can't max the utilization of cpu
    while i < len(process_pool) - 4:
        process_pool[i].start()
        process_pool[i + 1].start()
        process_pool[i + 2].start()
        process_pool[i + 3].start()
        process_pool[i + 4].start()
        process_pool[i].join()
        process_pool[i + 1].join()
        process_pool[i + 2].join()
        process_pool[i + 3].join()
        process_pool[i + 4].join()
        i += 5
        if i % 10 == 0:
            print("%d/%d" % (i, len(process_pool)))


if __name__ == '__main__':
    main()
