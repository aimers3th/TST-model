class Config:
    # Model params
    input_dim = 10
    d_model = 512
    nhead = 8
    num_layers = 3
    dropout = 0.5
    
    # Training params
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    model_path = "trained_model.pth"
