from src.dataset import ToolTrackingDataset
from src.model import TwoStreamTCN
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

source_folder = "./tool-tracking-data/"

dataset = ToolTrackingDataset(
    source_path=source_folder,
    tool_name="electric_screwdriver",
    window_length=0.5,
    overlap=0.25,
    exclude_time=True
)

# print("Feature Shape", dataset[1000][0].shape)
print("Target", dataset[0][-1])

d = {}

for i in range(len(dataset)):
    y = dataset[i][-1].item()
    v = d.setdefault(y, 0)
    v += 1
    d[y] = v


print("---------------------------")
print(d)

for i, j in enumerate(d.keys()):
    d[j] = i

print("---------------------------")
print(d)

model = TwoStreamTCN(num_classes=8)

model.eval()
x_acc, x_gyr, x_mag, x_mic, labels = dataset[0]
x_acc = x_acc.unsqueeze(0)  # Shape becomes [1, 3, 82]
x_gyr = x_gyr.unsqueeze(0)  # Shape becomes [1, 3, 82]
x_mag = x_mag.unsqueeze(0)  # Shape becomes [1, 3, 124]
x_mic = x_mic.unsqueeze(0)# Shape becomes [1, 1, 6400]
print("Input shapes:", x_acc.shape, x_gyr.shape, x_mag.shape, x_mic.shape)

with torch.no_grad():
    output = model(x_acc, x_gyr, x_mag, x_mic)

print("Model output:", output)
print("Model output shape:", output.shape)

# exit()

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()



def evaluate(model, loader, criterion, device, label_map):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    
    with torch.no_grad():
        for x_acc, x_gyr, x_mag, x_mic, labels in loader:
            x_acc = x_acc.to(device).float()
            x_gyr = x_gyr.to(device).float()
            x_mag = x_mag.to(device).float()
            x_mic = x_mic.to(device).float()
            
            # --- APPLY LABEL MAPPING ---
            # Transform raw tensor labels to indices using the dict 'd'
            labels_mapped = [label_map[int(l)] for l in labels]
            labels = torch.tensor(labels_mapped, dtype=torch.long).to(device)
            
            outputs = model(x_acc, x_gyr, x_mag, x_mic)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total, loss_sum / len(loader)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for x_acc, x_gyr, x_mag, x_mic, labels in loop:
        # Move to Device and ensure Float32 (Batch, Chan, Time)
        # Your dataset returns (Chan, Time) numpy arrays, DataLoader stacks to (Batch, Chan, Time)
        x_acc = x_acc.to(DEVICE).float()
        x_gyr = x_gyr.to(DEVICE).float()
        x_mag = x_mag.to(DEVICE).float()
        x_mic = x_mic.to(DEVICE).float()

        labels_mapped = [d[int(l)] for l in labels]
        labels = torch.tensor(labels_mapped, dtype=torch.long).to(DEVICE)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(x_acc, x_gyr, x_mag, x_mic)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
    epoch_acc = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    val_acc, val_loss = evaluate(model, test_loader, criterion, DEVICE, d)
    print(f"Epoch {epoch+1} Results -> Train Acc: {epoch_acc:.2f}% | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

print("[INFO] Training Complete.")