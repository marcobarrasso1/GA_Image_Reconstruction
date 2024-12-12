import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument('--generation', type=int, default=10000)
    parser.add_argument('--p_cross', type=float, default=0.8)
    parser.add_argument('--p_mut', type=float, default=0.1)
    parser.add_argument('--n_elites', type=int, default=3)
    parser.add_argument("--target_height", type=int, default=256)
    parser.add_argument('--target_width', type=int, default=256)
    parser.add_argument('--size', type=int, default=23)
    parser.add_argument("--shape", type=int, default=0)
    parser.add_argument("--gif", type=str, default="")
    args = parser.parse_args()
    return args