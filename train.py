#! /usr/bin/env python

import torch
from torch.utils.data import DataLoader
from models import PathPlanningModel

import utils
import pdb

if __name__ == "__main__":
    data = torch.load("train.pt")
    data = utils.PathPlanningDataset(data)

    train_loader = DataLoader(
        data, batch_size=32, collate_fn=utils.collate_fn, shuffle=True
    )

    # Model instantiation
    model = PathPlanningModel()

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2

    for epoch in range(num_epochs):
        for batch in train_loader:
            batch_map, batch_start, batch_goal, batch_path = batch

            # Forward pass
        #     optimizer.zero_grad()
        #     outputs = model(batch_map, batch_start, batch_goal, batch_lengths)

        #     # Compute loss (only use the non-padded parts of the sequences)
        #     packed_outputs = pack_padded_sequence(
        #         outputs, batch_lengths, batch_first=True, enforce_sorted=False
        #     )
        #     packed_targets = pack_padded_sequence(
        #         batch_path, batch_lengths, batch_first=True, enforce_sorted=False
        #     )
        #     loss = criterion(packed_outputs.data, packed_targets.data)

        #     # Backward pass and optimize
        #     loss.backward()
        #     optimizer.step()

        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
