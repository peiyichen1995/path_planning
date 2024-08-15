#! /usr/bin/env python

import torch
from models import PathPlanningModel
import os
import utils
import pdb

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = utils.data_loader(
        "train.pt", batch_size=500, data_size=1000, device=device
    )
    validation_loader = utils.data_loader(
        "validation.pt", batch_size=500, data_size=500, device=device
    )

    # Model instantiation
    model = PathPlanningModel().to(device)
    cp_dir = "model.pt"
    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 200
    npath = 300
    best_loss = float("Inf")

    for epoch in range(num_epochs):
        for batch in train_loader:
            batch_map, batch_start, batch_goal, batch_path = batch

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_map.unsqueeze(1), batch_start, batch_goal, npath)
            loss = criterion(outputs, batch_path)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        validation_loss = 0
        for batch in validation_loader:
            batch_map, batch_start, batch_goal, batch_path = batch

            # Forward pass
            with torch.no_grad():
                outputs = model(batch_map.unsqueeze(1), batch_start, batch_goal, npath)
                validation_loss += criterion(outputs, batch_path).item()
            validation_loss /= len(validation_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {validation_loss:.4f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            if os.path.exists(cp_dir):
                os.remove(cp_dir)
                print("Remove previous model.")
            print("Save updated model.")
            torch.save(model.state_dict(), cp_dir)
