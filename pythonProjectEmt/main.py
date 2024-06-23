import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from Dataloader import train_loader, val_loader

class SimpleNN(nn.Module):
    def __init__(self, input_size, dropout_prob):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)  # Increased size
        self.fc2 = nn.Linear(1024, 512)  # Increased size
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2)  # 2 output classes: whale and sparrow
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer with variable probability

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    train_losses = []
    val_losses = []
    dropout_probs = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.view(inputs.size(0), -1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}%")

        # Append losses for plotting
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        dropout_probs.append(model.dropout.p)  # Record the current dropout probability

        # Step the learning rate scheduler
        scheduler.step()

        # Check for overfitting condition after epoch 10
        if epoch >= 9 and val_losses[-1] > val_losses[-2]:
            print("Overfitting detected!")
            break

    return train_losses, val_losses, dropout_probs, accuracy

# Hyperparameters
input_size = 224 * 224 * 3  # Adjust based on image size and channels
dropout_prob = 0.5  # Initial dropout probability
num_epochs = 20

# Instantiate the model
model = SimpleNN(input_size, dropout_prob)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

# Train the model
train_losses, val_losses, dropout_probs, final_accuracy = train_model(model, criterion, optimizer, scheduler, num_epochs)

# Plotting the training and validation losses
plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)

# Plot dropout probability
plt.subplot(1, 2, 2)
plt.plot(epochs, dropout_probs, marker='o', linestyle='-', color='r')
plt.xlabel('Epochs')
plt.ylabel('Dropout Probability')
plt.title('Dropout Probability over Epochs')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final validation accuracy
print(f"Final Validation Accuracy: {final_accuracy}%")
