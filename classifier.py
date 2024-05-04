import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from transformers import pipeline

class multiclass_classifier(nn.Module):
    """
    simple 3 layer neural network (2 hidden layer)
    """
    def __init__(self, input_size, num_classes, hidden_size1 = 128, hidden_size2 = 64):
        super(multiclass_classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

def get_NLP_zeroshot_model():
    """
    see https://huggingface.co/facebook/bart-large-mnli
    loading a zeroshot text classification model
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier


def preprocess_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True)


def classifier_fit(model, X_tensor:torch.Tensor, y_tensor:torch.Tensor, device="cuda", num_epochs=5, batch_size=64):

    # move model to device
    model = model.to(device)

    # create PyTorch dataset and data loaders
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate epoch loss and training accuracy
        epoch_loss /= len(train_loader)
        train_accuracy = correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    return model
