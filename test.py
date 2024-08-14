#! /usr/bin/env python

import torch
from models import PathPlanningModel
import utils

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PathPlanningModel().to(device)
    cp_dir = "model.pt"

    model.load_state_dict(torch.load(cp_dir))

    validation_loader = utils.data_loader("validation.pt", 500, device, 500)
    criterion = torch.nn.MSELoss()

    validation_loss = 0
    npath = 300
    for batch in validation_loader:
        batch_map, batch_start, batch_goal, batch_path = batch

        # Forward pass
        with torch.no_grad():
            outputs = model(batch_map.unsqueeze(1), batch_start, batch_goal, npath)
            validation_loss += criterion(outputs, batch_path).item()
    validation_loss /= len(validation_loader)
    print(f"Loss: {validation_loss:.4f}")
