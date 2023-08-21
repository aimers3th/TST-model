import torch
from config import Config
from model import TransformerModel
from utils import evaluate
from data_loader import get_data_loaders, normalize_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    # TODO: Load your raw data here for evaluation
    raw_data = ...
    target = ...

    data, _ = normalize_data(raw_data)  # Assuming evaluation data uses the same normalization
    _, test_loader = get_data_loaders(data, target, Config.seq_length, Config.batch_size)

    model = TransformerModel(input_dim=Config.input_dim, d_model=Config.d_model, nhead=Config.nhead, num_layers=Config.num_layers, dropout=Config.dropout)
    model.load_state_dict(torch.load(Config.model_path))
    model.to(device)

    criterion = torch.nn.MSELoss()
    eval_loss = evaluate(model, criterion, test_loader, device)
    print(f"Eval loss: {eval_loss:.4f}")

if __name__ == "__main__":
    main()
