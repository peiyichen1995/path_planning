import torch
import torch.nn as nn
import pdb


class PathPlanningModel(nn.Module):
    def __init__(self):
        super(PathPlanningModel, self).__init__()

        # Map Encoder: Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers for Start and Goal Points
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)

        # Fully Connected Layers after concatenation
        self.fc3 = nn.Linear(
            128 * 12 * 12 + 128, 256
        )  # Adjust the input size accordingly
        self.fc4 = nn.Linear(256, 512)

        # LSTM Layer for path prediction
        self.lstm = nn.LSTM(512, 256, batch_first=True)

        # Output Layer
        self.fc5 = nn.Linear(256, 2)

    def forward(self, map_input, start_input, goal_input):
        pdb.set_trace()
        # Map encoding
        x = self.pool(torch.relu(self.conv1(map_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the output

        # Start and Goal encoding
        y = torch.cat((start_input, goal_input), dim=1)
        y = torch.relu(self.fc1(y))
        y = torch.relu(self.fc2(y))

        # Concatenate map features with start and goal features
        z = torch.cat((x, y), dim=1)
        z = torch.relu(self.fc3(z))
        z = torch.relu(self.fc4(z))

        # Reshape for LSTM
        z = z.unsqueeze(1).repeat(1, map_input.size(2), 1)  # Repeat for each timestep

        # LSTM for path prediction
        z, _ = self.lstm(z)

        # Predict the path
        path_output = self.fc5(z)

        return path_output
