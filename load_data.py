#! /usr/bin/env python

import torch
import argparse
import os

from utils import MapDataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-D",
        "--data",
        choices=["train", "validation", "test"],
    )

    args = parser.parse_args()
    if not args.data:
        raise Exception("No data folder specified!")

    dataset = MapDataset(args.data)
    ndata = len(dataset)
    sample, _ = dataset[0]
    map, start, end, path = sample.numpy()
    h, w = map.shape
    ncoords = 2
    npath = 300
    maps = torch.empty(ndata, h, w)
    starts = torch.empty(ndata, ncoords)
    ends = torch.empty(ndata, ncoords)
    paths = []
    for i in range(ndata):
        sample, _ = dataset[i]
        map, start, end, path = sample.numpy()
        maps[i] = torch.Tensor(map)
        starts[i] = torch.Tensor(start)
        ends[i] = torch.Tensor(end)
        paths.append(torch.Tensor(path))

    # pdb.set_trace()
    paths[0] = torch.nn.ConstantPad2d((0, 0, 0, npath - paths[0].shape[0]), 0.0)(
        paths[0]
    )
    paths = torch.nn.utils.rnn.pad_sequence(paths, batch_first=True)
    data = {"maps": maps, "starts": starts, "ends": ends, "paths": paths}
    cp_dir = args.data + ".pt"
    if os.path.exists(cp_dir):
        os.remove(cp_dir)

    torch.save(data, cp_dir)
