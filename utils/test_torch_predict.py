import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set device to MPS if available, otherwise CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = SimpleNN().to(device)

# Load the trained model
model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
model.eval()

# Example prediction on a single image
# Get a single batch from the test loader
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Take one image from the batch
image = images[2].unsqueeze(0).to(device)
label = labels[2].item()

# Make a prediction
with torch.no_grad():
    output = model(image)
    _, predicted_label = torch.max(output, 1)

predicted_label = predicted_label.item()

# Display the image and prediction
plt.imshow(image.cpu().squeeze(), cmap='gray')
plt.title(f'Predicted: {predicted_label}, Actual: {label}')
plt.show()
