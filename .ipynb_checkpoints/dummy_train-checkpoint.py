import torch
import torch.nn as nn
import torch.optim as optim
import synth_dataset as sd  # Import the synthetic dataset
from tqdm import tqdm
import os

class DummyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        # No activation in the final layer for regression

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


def train_model(model, dataloader, criterion, optimizer, num_epochs=10, checkpoint_dir=None):
    """Train the model for a specified number of epochs"""
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader):
            # Move data to the correct device (CPU or GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        # Save the model after each epoch
        if checkpoint_dir is not None:
            # check if the directory exists, otherwise create it
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch{epoch+1}.pth")
    print("Finished Training")



if __name__ == '__main__':
    # Hyperparameters
    num_samples = 10000
    img_height = 480
    img_width = 640
    channels = 3
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    num_workers = 12
    checkpoint_directory = "checkpoints"

    # Create dataset and dataloader
    dataset = sd.SyntheticDataset(num_samples, img_height, img_width, channels)
    dataloader = sd.create_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Create the dummy model
    model = DummyModel(in_channels=channels, out_channels=1)

    # Choose a loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Check if CUDA (GPU) is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device) # Move the model to the device

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=num_epochs, checkpoint_dir=checkpoint_directory)

    # --- Evaluation (after training) ---
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        for images, labels in dataloader:  # Use a few batches for evaluation
             images = images.to(device)
             labels = labels.to(device)
             outputs = model(images)

             # Apply sigmoid to get probabilities (since we used BCEWithLogitsLoss)
             predictions = torch.sigmoid(outputs)

             # --- Visualization ---
             import matplotlib.pyplot as plt
             for i in range(images.size(0)):  # Iterate through the batch
                  image_np = images[i].cpu().permute(1, 2, 0).numpy()
                  label_np = labels[i].cpu().squeeze().numpy()  # Remove channel dim
                  prediction_np = predictions[i].cpu().squeeze().numpy()

                  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                  axes[0].imshow(image_np)
                  axes[0].set_title("Input Image")

                  axes[1].imshow(label_np, cmap='gray')
                  axes[1].set_title("Ground Truth Label")

                  axes[2].imshow(prediction_np, cmap='gray')  # Show the predicted mask
                  axes[2].set_title("Prediction")
                  plt.show()
                  break  

                    