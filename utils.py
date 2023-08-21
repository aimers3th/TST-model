import torch

def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    total_loss = 0.
    for batch in data_loader:
        data, target = batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch in data_loader:
            data, target = batch
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(data_loader)
