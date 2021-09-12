import os
import argparse

import multiprocessing as multi

from data.shinra_utils import ShinraData, ShirnaSystemData

def load_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jp5_path", type=str, required=True)
    parser.add_argument("--train_location_path", type=str, required=True)
    parser.add_argument("--train_organization_path", type=str, required=True)

    parser.add_argument("--system_results_jp5_path", type=str, required=True)
    parser.add_argument("--system_results_location_path", type=str, required=True)
    parser.add_argument("--system_results_organization_path", type=str, required=True)

    parser.add_argument("--roberta_dir", type=str, required=True)
    parser.add_argument("--bpe_codes_name", type=str, default="codes.txt")
    parser.add_argument("--vocab_name", type=str, default="vocab.json")

    parser.add_argument("--sampling_size", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=1234)

    parser.add_argument("--output_dir", type=str, default="./dataset")

    parser.add_argument("--num_workers", type=int, default=multi.cpu_count())
    return parser.parse_args()

if __name__ == '__main__':
    args = load_arg()

    bpe_codes_path = os.path.join(
        args.roberta_dir,
        args.bpe_codes_name
    )
    vocab_path = os.path.join(
        args.roberta_dir,
        args.vocab_name
    )
    tasks = [
        ("Location", args.train_location_path, args.system_results_location_path),
        ("Organization", args.train_organization_path, args.system_results_organization_path),
        ("JP-5", args.train_jp5_path, args.system_results_jp5_path),
    ]

    for name, train_path, system_results_path in tasks:
        data = ShinraData(
            train_path,
            bpe_codes_path,
            vocab_path,
            args.num_workers
        )
        data.preprocess(
            output_dir = os.path.join(args.output_dir, "preprocessed_train", name)
        )
        train_pageids = data.get_pageids()

        data = ShirnaSystemData(
            train_path,
            system_results_path,
            bpe_codes_path,
            vocab_path,
            args.num_workers,
            sampling_size=args.sampling_size,
            random_seed=args.random_seed,
            train_pageids=train_pageids
        )
        if args.sampling_size is None:
            output_dir = os.path.join(args.output_dir, "preprocessed_system_results", name)
        else:
            output_dir = os.path.join(args.output_dir, f"preprocessed_system_results_{args.sampling_size}", name)
        data.preprocess(output_dir=output_dir)
        data.save_systems(output_dir=output_dir)
