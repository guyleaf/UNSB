import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate result by FID and Sort.",
    )
    parser.add_argument(
        "images_dir",
        type=str,
        help="The images folder for comparison, e.g. ./datasets/horse2zebra/testB.",
    )
    parser.add_argument(
        "result_dir", type=str, help="The result folder of the experiment."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Which epochs are to be evaluated. By default, evaluate all epochs.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="test",
        help="Which phase are to be evaluated.",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="asc",
        choices=["asc", "desc"],
        help="Which order of metric value is used in sorting.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Show top-k epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes to use for data loading.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. Like cuda, cuda:0 or cpu",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
        help=(
            "Dimensionality of Inception features to use. "
            "By default, uses pool3 features"
        ),
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Not to load the existed fid results from the result folder",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Not to save fid results in the result folder",
    )
    args = parser.parse_args()
    assert os.path.isdir(args.images_dir) and os.path.isdir(args.result_dir)
    assert args.topk > 0
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)
    num_workers = args.num_workers

    result_dir = Path(args.result_dir)
    if args.epochs is None:
        epoch_dirs = list(result_dir.glob(f"{args.phase}_*"))
    else:
        epoch_dirs = [result_dir / f"{args.phase}_{epoch}" for epoch in args.epochs]

    results = defaultdict(dict)

    # load previous calculated results
    fid_json = result_dir / "fid.json"
    if not args.no_load and fid_json.exists():
        with open(fid_json) as f:
            results.update(json.load(f))

    for epoch_dir in tqdm(epoch_dirs):
        # skip calculated epochs
        if epoch_dir.name in results:
            continue

        fake_dir = epoch_dir / "images"
        fake_dirs = sorted(fake_dir.glob("fake_*"))

        for fake in fake_dirs:
            fid = calculate_fid_given_paths(
                [args.images_dir, fake.as_posix()],
                args.batch_size,
                device,
                args.dims,
                num_workers,
            )
            results[epoch_dir.name][fake.name] = fid

    topk_results = sorted(
        results.items(),
        key=lambda v: min(v[1].values()),
        reverse=(args.order == "desc"),
    )[: args.topk]

    pprint(dict(topk_results), sort_dicts=False)

    if not args.no_save:
        with open(fid_json, "w") as f:
            json.dump(results, f, indent=4)
