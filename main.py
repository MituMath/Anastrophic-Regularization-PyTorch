import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import torch.fft
import torch.nn.functional as F

# --- 1. DATA PREPARATION (Split-MNIST) ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def get_split_dataloaders(dataset, classes, batch_size=64):
    indices = [i for i, target in enumerate(dataset.targets) if target in classes]
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

# Task A: Digits 0-4
train_loader_A = get_split_dataloaders(trainset, [0, 1, 2, 3, 4])
test_loader_A = get_split_dataloaders(testset, [0, 1, 2, 3, 4])

# Task B: Digits 5-9
train_loader_B = get_split_dataloaders(trainset, [5, 6, 7, 8, 9])
test_loader_B = get_split_dataloaders(testset, [5, 6, 7, 8, 9])

# --- 2. NEURAL NETWORK ARCHITECTURE (CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- 3. ANASTROPHIC REGULARIZATION ---
class AnastrophicRegularizer(nn.Module):
    def __init__(self, lambda_reg=1.0, eta_reg=3.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.eta_reg = eta_reg

    def compute_phi(self, w):
        """Calculates Spectral Coherence (Phi) via 1D FFT."""
        fft_w = torch.fft.fft(w.view(-1))
        amplitudes = torch.abs(fft_w)
        phases = torch.angle(fft_w)
        
        p_j = (amplitudes ** 2) / (torch.sum(amplitudes ** 2) + 1e-8)
        complex_sum = torch.sum(p_j * torch.exp(1j * phases))
        
        return torch.abs(complex_sum)

    def compute_beta_proxy(self, w, w_prev):
        """Continuous proxy for Anastrophic Beta (BB) measuring harmonic tension."""
        fft_w = torch.fft.fft(w.view(-1))
        fft_prev = torch.fft.fft(w_prev.view(-1))
        
        complex_w = torch.view_as_real(fft_w)
        complex_prev = torch.view_as_real(fft_prev)
        
        return F.mse_loss(complex_w, complex_prev)

    def forward(self, model, model_prev):
        loss_ana = 0.0
        for (name, param), (name_prev, param_prev) in zip(model.named_parameters(), model_prev.named_parameters()):
            if 'weight' in name:
                phi = self.compute_phi(param)
                # .detach() is critical to anchor the previous structural state
                beta = self.compute_beta_proxy(param, param_prev.detach())
                
                loss_ana += self.lambda_reg * (1 - phi) + self.eta_reg * beta
                
        return loss_ana

# --- 4. PHASE 1: TRAINING TASK A ---
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("--- Starting Task A (Digits 0-4) Training ---")
model.train()
for epoch in range(3):
    for images, labels in train_loader_A:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

acc_A = evaluate_accuracy(model, test_loader_A)
print(f"Accuracy on Task A after training Task A: {acc_A:.2f}%\n")

# Freeze the base model to preserve its structural return invariants
model_A_frozen = copy.deepcopy(model)
model_A_frozen.eval()

# --- 5. PHASE 2: TRAINING TASK B WITH ANASTROPHIC REGULARIZATION ---
regularizer = AnastrophicRegularizer(lambda_reg=1.0, eta_reg=3.0) 

print("--- Starting Task B (Digits 5-9) Training with R_ana ---")
optimizer_B = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    running_loss_class = 0.0
    running_loss_ana = 0.0
    
    for images, labels in train_loader_B:
        optimizer_B.zero_grad()
        outputs = model(images)
        
        loss_class = criterion(outputs, labels)
        
        # Apply Anastrophic Theory to preserve structural relationships
        loss_ana = regularizer(model, model_A_frozen)
        
        loss = loss_class + loss_ana 
        loss.backward()
        optimizer_B.step()
        
        running_loss_class += loss_class.item()
        running_loss_ana += loss_ana.item()
        
    print(f"Epoch {epoch+1} | Classification Loss: {running_loss_class/len(train_loader_B):.4f} | R_ana Loss: {running_loss_ana/len(train_loader_B):.4f}")

# --- 6. FINAL EVALUATION ---
print("\n--- Final Results ---")
acc_B_final = evaluate_accuracy(model, test_loader_B)
acc_A_final = evaluate_accuracy(model, test_loader_A)

print(f"Accuracy on NEW Task B (5-9): {acc_B_final:.2f}%")
print(f"RETAINED Accuracy on Task A (0-4): {acc_A_final:.2f}%")
