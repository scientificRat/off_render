from render import Render
import os
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


def main():
    # traversal the directors
    sub_dirs = get_immediate_subdirectories(MODEL_NET_ROOT)
    process_pool = mp.Pool(min(os.cpu_count() - 1, 20))
    render = Render()
    for sub_dir in sub_dirs:
        print("dealing " + sub_dir + "...")
        # make output dirs
        out_sub_dir = OUTPUT_DIR + "/" + sub_dir
        make_dir_not_exist(out_sub_dir + "/test")
        make_dir_not_exist(out_sub_dir + "/train")
        # ready to convert
        source_dir = MODEL_NET_ROOT + "/" + sub_dir
        for d in ["/test", "/train"]:
            print(d)
            curr_dir = source_dir + d
            off_files = list(get_off_file_in_dir(curr_dir))
            for i, off_file in enumerate(off_files):
                render.render_and_save(curr_dir + "/" + off_file, out_sub_dir + d)
                if i % 10 == 0:
                    print("%d/%d" % (i, len(off_files)))


if __name__ == '__main__':
    main()
