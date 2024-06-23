# Define the model class without Dropout
import torch
from matplotlib import pyplot as plt
from torch import optim, nn

from Dataloader import train_loader, val_loader
from main import num_epochs, input_size, criterion


class SimpleNNWithoutDropout(nn.Module):
    def __init__(self, input_size):
        super(SimpleNNWithoutDropout, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)  # 2 output classes: whale and sparrow

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Instantiate the model without Dropout
model_without_dropout = SimpleNNWithoutDropout(input_size)

# Define loss function and optimizer
optimizer = optim.Adam(model_without_dropout.parameters(), lr=0.001)

# Training loop without Dropout
train_losses_without_dropout = []
val_losses_without_dropout = []
train_accuracies_without_dropout = []
val_accuracies_without_dropout = []
for epoch in range(num_epochs):
    model_without_dropout.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        outputs = model_without_dropout(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses_without_dropout.append(train_loss / len(train_loader))
    train_accuracy = 100 * correct / total
    train_accuracies_without_dropout.append(train_accuracy)

    # Validation step
    model_without_dropout.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.view(inputs.size(0), -1)

            outputs = model_without_dropout(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses_without_dropout.append(val_loss / len(val_loader))
    val_accuracy = 100 * correct / total
    val_accuracies_without_dropout.append(val_accuracy)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Train Accuracy: {train_accuracy}%, Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}%")

# Plotting the training and validation losses without Dropout
plt.figure(figsize=(12, 5))
epochs = range(1, num_epochs + 1)
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses_without_dropout, label='Train Loss')
plt.plot(epochs, val_losses_without_dropout, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss without Dropout')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies_without_dropout, label='Train Accuracy')
plt.plot(epochs, val_accuracies_without_dropout, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy without Dropout')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final validation accuracy without Dropout
final_val_accuracy_without_dropout = val_accuracies_without_dropout[-1]
print(f"Final Validation Accuracy without Dropout: {final_val_accuracy_without_dropout}%")
