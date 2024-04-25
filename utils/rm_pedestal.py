import argparse, os

import numpy as np

def main(IN_DIR, OUT_DIR):
    A_mins = []
    A_mins_nonzero = []
    dir_len = len(list(os.scandir(IN_DIR)))
    for i, entry in enumerate(os.scandir(IN_DIR)):
        print("[{}/{}]".format(i, dir_len), end='\r')
        if entry.name.endswith('.npy'):
            AB = np.load(entry.path)
            _, _, w = AB.shape
            w2 = int(w/2)
            A = AB[0, :, :w2]

            A[A != 0] -= 74.0 # ND pedestal

            A_mins.append(A.min())
            A_mins_nonzero.append(A[A != 0].min())
            
            np.save(os.path.join(OUT_DIR, entry.name), AB)            
            
    print("[{}/{}]".format(dir_len, dir_len))
    print("minA = {}".format(min(A_mins)))
    print("minA nonzero = {}".format(min(A_mins_nonzero)))

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("in_dir")
    parser.add_argument("out_dir")

    args = parser.parse_args()

    return (args.in_dir, args.out_dir)

if __name__ == '__main__':
    arguments = parse_arguments()

    main(*arguments)
