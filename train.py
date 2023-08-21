import torch
from config import Config
from model import TransformerModel
from utils import train_one_epoch, evaluate
from data_loader import get_data_loaders, normalize_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    # TODO: Load your raw data here
    raw_data = ...
    target = ...

    data, scaler = normalize_data(raw_data)
    train_loader, test_loader = get_data_loaders(data, target, Config.seq_length, Config.batch_size)

    model = TransformerModel(input_dim=Config.input_dim, d_model=Config.d_model, nhead=Config.nhead, num_layers=Config.num_layers, dropout=Config.dropout)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    for epoch in range(Config.num_epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        eval_loss = evaluate(model, criterion, test_loader, device)
        print(f"Epoch {epoch+1}/{Config.num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), Config.model_path)

if __name__ == "__main__":
    main()
