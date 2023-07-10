import argparse

from src.data_sampling.random_sampling import *


def main():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--path', required=False, help='path', default="data/experiment_2/test.csv")
    parser.add_argument('--random_sample_size', required=False, help='path', default=50)
    parser.add_argument('--huggingface_repo', required=False, help='huggingface_repo', default='/content/experiment_1')

    # sampling steps
    # 1 - generate random author names for each set
    # 2 - generate dataset split indexes
    parser.add_argument('--sampling_step', required=True, help='sampling step', default=1)

    args = parser.parse_args()

    if int(args.sampling_step) == 1:
        generate_random_samples(int(args.random_sample_size), args.path)
    else:
        generate_dataset_split_indexes(int(args.random_sample_size), args.path, args.huggingface_repo)


if __name__ == "__main__":
    main()
