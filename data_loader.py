import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, seq_length):
        self.data = data
        self.target = target
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (self.data[index:index+self.seq_length], self.target[index+self.seq_length])

def get_data_loaders(data, target, seq_length, batch_size):
    dataset = TimeSeriesDataset(data, target, seq_length)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def normalize_data(data):
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    return torch.tensor(data_normalized, dtype=torch.float32), scaler
