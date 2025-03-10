import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
import synth_dataset as sd 
import os

class KeypointDetector(nn.Module):
    """
    A neural network model for keypoint detection. The model uses MobileNetV3 as an encoder to extract image features and a custom decoder to generate a keypoint heatmap.
    args:
        pretrained (bool): if True, initializes the encoder (MobileNetV3) with pre-trained weights. default is True.
    """
    
    def __init__(self, pretrained=True):
        super(KeypointDetector, self).__init__()
        
        # loading MobileNetV3
        mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
        self.encoder = mobilenet.features  # use the feature layers of MobileNetV3

        # decoder 
        self.decoder = nn.Sequential(
            # first Conv layer to reduce the channels from 960 (MobileNet output) to 256
            nn.Conv2d(960, 256, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  # ReLU activation for non-linearity
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # upsample the feature map
            
            # second Conv layer to reduce the channels to 128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # another upsampling
            
            # third Conv layer to reduce the channels to 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # final Conv layer to get the output in a single channel (for binary keypoint heatmap)
            nn.Conv2d(64, 1, kernel_size=1),  
            # sigmoid to output values between 0 and 1
            nn.Sigmoid()  
        )

    def forward(self, x):
        """
        Forward pass for the keypoint detector.

        args:
            x (Tensor): input tensor of shape (batch_size, channels, height, width).

        returns:
            Tensor: the generated keypoint heatmap of shape (batch_size, 1, height, width).
        """
        # extract features using the MobileNetV3 encoder
        features = self.encoder(x)
        
        # pass the extracted features through the decoder to generate keypoint heatmap
        keypoint_heatmap = self.decoder(features)
        
        # upsample the keypoint heatmap to match the target size (640x480), which is the expected resolution
        keypoint_heatmap = F.interpolate(keypoint_heatmap, size=(640, 480), mode='bilinear', align_corners=False)
        
        return keypoint_heatmap


def train_model(model, dataloader, criterion, optimizer, num_epochs=10, checkpoint_dir=None):
    """Train the model for a specified number of epochs"""
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(images)  # Forward pass
            loss = criterion(outputs.squeeze(1), labels)  # Compute loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Save the model checkpoint
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch{epoch+1}.pth")

    print("Finished Training")


if __name__ == '__main__': 
    # hyperparameters
    num_samples = 100  
    img_height, img_width = 480, 640 
    channels = 3  
    batch_size = 16  
    learning_rate = 0.0001  
    num_epochs = 5 
    num_workers = 4  
    checkpoint_directory = "checkpoints"  

    #synthetic dataset and dataloader for training
    dataset = sd.SyntheticDataset(num_samples, img_height, img_width, channels)
    dataloader = sd.create_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

    # keypoint detection model
    model = KeypointDetector()

    # loss function and optimizer
    criterion = nn.BCELoss()  # binary cross-entropy loss for keypoint heatmap
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer for weight updates

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    train_model(model, dataloader, criterion, optimizer, num_epochs=num_epochs, checkpoint_dir=checkpoint_directory)

    # --- evaluation & visualization ---
    model.eval()  
    
    with torch.no_grad():  
        for images, labels in dataloader:  
            images = images.to(device)
            labels = labels.to(device)
            
            # get model outputs (predictions)
            outputs = model(images)
            
            # print some statistics about the output (min, max, mean values)
            print(outputs.min(), outputs.max(), outputs.mean())


            predictions = outputs.cpu().squeeze().numpy()
            for i in range(images.size(0)):
                image_np = images[i].cpu().permute(1, 2, 0).numpy()
                label_np = labels[i].cpu().squeeze().numpy()
                prediction_np = predictions[i]

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(image_np)
                axes[0].set_title("Input Image")

                axes[1].imshow(label_np, cmap='hot')
                axes[1].set_title("Ground Truth Keypoints")

                axes[2].imshow(prediction_np, cmap='hot')
                axes[2].set_title("Predicted Keypoints")
                plt.show()  
                
                break  
