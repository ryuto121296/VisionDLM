import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os

# --- Configuration ---
# Based on user input:
MODEL_BACKBONE = "resnet18"
NUM_CLASSES = 2  # Assuming binary classification (Good/Bad)
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10
DATA_DIR = "../models/" # Assuming data is relative to the script location

# --- 1. Model Initialization (Using pre-trained weights) ---
def initialize_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Initializes a pre-trained model from torchvision and adapts the final layer.
    """
    print(f"Loading pre-trained {model_name}...")
    # Load the pre-trained model
    model = models.__getattr__(model_name)(pretrained=True)
    
    # Freeze all the parameters initially (optional, but good practice for fine-tuning)
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final classification layer (the head)
    # ResNet typically has an output layer that needs replacement for custom classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Unfreeze the new layer for training
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model

# --- 2. Data Loading (Placeholder - Adapt this based on actual dataset structure) ---
def get_data_loaders(data_dir: str, batch_size: int):
    """
    Placeholder function to load datasets. 
    This must be adapted to match the data loading logic from existing notebooks.
    """
    print("--- WARNING: Using placeholder data loading. Adapt this function! ---")
    # Example transformation pipeline (adjust as necessary)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # In a real scenario, you would use ImageFolder or a custom Dataset class here.
    # For now, we return dummy loaders to allow the script structure to be sound.
    dummy_dataset = torch.utils.data.Dataset()
    dummy_dataset.__len__ = lambda: 100 # Simulate 100 images
    dummy_dataset.__getitem__ = lambda idx: (torch.randn(3, 224, 224), torch.tensor(0))
    
    train_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# --- 3. Training Function ---
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """
    Trains the model for a specified number of epochs.
    """
    print("Starting model training...")
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / (len(train_loader.dataset) * len(train_loader))
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy for validation
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss_avg = val_loss / (len(val_loader.dataset) * len(val_loader))
        val_accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_accuracy:.2f}%")

# --- 4. Inference and Thresholding (Addressing 100% Recall Goal) ---
def apply_thresholding(model: nn.Module, data_loader: DataLoader, device: torch.device, threshold_good: float, threshold_bad: float):
    """
    Performs inference and applies custom thresholding logic to meet recall goals.
    """
    print("\n--- Applying Custom Thresholding Logic ---")
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Assuming the output layer is logits, we apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # For binary classification (Good/Bad), we look at the probability of the 'Good' class (index 0)
            # Assuming index 0 is 'Good' and index 1 is 'Bad' based on typical setup.
            # We take the probability of the first class (index 0) as the 'Good' score.
            good_scores = probabilities[:, 0].cpu().numpy()
            all_scores.extend(good_scores)

    print(f"Inference complete. Processed {len(all_scores)} samples.")
    
    # Applying the user-defined logic:
    # Good if score >= threshold_good (0.95)
    # Bad if score <= threshold_bad (0.05)
    # Otherwise, classify as 'Bad' if score < 0.7 (as per user request)
    
    results = []
    for score in all_scores:
        if score >= threshold_good:
            result = "Good"
        elif score <= threshold_bad:
            result = "Bad"
        elif score < 0.7:
            result = "Bad" # Explicitly setting this based on user's fallback logic
        else:
            result = "Uncertain/Review" # Or handle as needed
        results.append(result)
        
    print(f"Example results based on thresholding: {results[:5]}...")
    return results


# --- Main Execution ---
if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = initialize_model(MODEL_BACKBONE, NUM_CLASSES).to(device)

    # 2. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Load Data
    train_loader, val_loader = get_data_loaders(DATA_DIR, BATCH_SIZE)

    # 4. Train Model
    train_model(model, train_loader, val_loader, criterion, optimizer, device)

    # 5. Test Thresholding Logic (Using validation set as a proxy for testing)
    # NOTE: The actual thresholding logic should be applied to the final test set.
    apply_thresholding(model, val_loader, device, threshold_good=0.95, threshold_bad=0.05)