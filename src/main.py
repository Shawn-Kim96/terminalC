import os
import sys
import pandas as pd
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.utils import class_weight
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Add project path to system path

path_splited = os.path.abspath('.').split('rsi_divergence_detector')[0]
PROJECT_PATH = os.path.join(path_splited, 'rsi_divergence_detector')
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from src.dataset.lstm_dataset import LSTMDivergenceDataset
from src.model.lstm_mixed import MixedLSTMModel
from src.training.train_lstm import train_model, evaluate_model, plot_results, model_naming
from src.model.transformer import TransformerMixedModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_device():
    """
    Selects the best available device: MPS (Apple GPU), CUDA (NVIDIA GPU), or CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")
    return device



def main_transformer():
    """
    Main function to train, evaluate, and test the LSTM model for divergence classification.
    """
    # Load data
    logger.info("Loading data...")
    df = pd.read_pickle(os.path.join(PROJECT_PATH, 'data', 'processed_data', 'training_data.pickle'))
    divergence_data = pd.read_pickle(os.path.join(PROJECT_PATH, 'data', 'processed_data', 'divergence_data.pickle'))

    # Filter 5-minute timeframe data
    price_df = df[df['timeframe'] == '5m'].copy()
    divergence_df = divergence_data['15m'].copy()  # Assuming '5m' key exists

    # Split divergence_df into train/val/test
    logger.info("Splitting data into train, validation, and test sets...")
    total_events = len(divergence_df)
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    train_count = int(total_events * train_ratio)
    val_count = int(total_events * val_ratio)
    test_count = total_events - train_count - val_count

    divergence_df_train = divergence_df.iloc[:train_count]
    divergence_df_val = divergence_df.iloc[train_count:train_count+val_count]
    divergence_df_test = divergence_df.iloc[train_count+val_count:]

    logger.info(f"Train events: {len(divergence_df_train)}, "
                f"Validation events: {len(divergence_df_val)}, "
                f"Test events: {len(divergence_df_test)}")

    # Prepare divergence_data for multiple timeframes (if applicable)
    # Assuming divergence_data contains multiple timeframes
    # Example: divergence_data = {'5m': ddf_5m, '15m': ddf_15m, '1h': ddf_1h, ...}
    # For simplicity, using only '5m' here
    divergence_data_subset = {'15m': divergence_df_train}

    # Initialize Dataset
    logger.info("Initializing datasets...")
    if "train_dataset.pickle" not in os.listdir(f"{PROJECT_PATH}/data/training_data"):

        train_dataset = LSTMDivergenceDataset(divergence_df=divergence_df_train, 
                                            price_df=price_df, 
                                            divergence_data=divergence_data, 
                                            seq_length=288)  # 288 * 5min = 24 hours
    else:
        train_dataset = pd.read_pickle(f"{PROJECT_PATH}/data/training_data/train_dataset.pickle")

    
    # Use the same scaler for validation and test
    scaler = train_dataset.scaler
    if "val_dataset.pickle" not in os.listdir(f"{PROJECT_PATH}/data/training_data"):

        val_dataset = LSTMDivergenceDataset(divergence_df=divergence_df_val, 
                                                price_df=price_df, 
                                                divergence_data=divergence_data, 
                                                seq_length=288, 
                                                scaler=scaler)
    else:
        val_dataset = pd.read_pickle(f"{PROJECT_PATH}/data/training_data/val_dataset.pickle")
    
    if "test_dataset.pickle" not in os.listdir(f"{PROJECT_PATH}/data/training_data"):

        test_dataset = LSTMDivergenceDataset(divergence_df=divergence_df_test, 
                                            price_df=price_df, 
                                            divergence_data=divergence_data, 
                                            seq_length=288, 
                                            scaler=scaler)
    else:
        test_dataset = pd.read_pickle(f"{PROJECT_PATH}/data/training_data/test_dataset.pickle")

    # Initialize DataLoaders
    logger.info("Creating DataLoaders...")
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Define device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize Transformer Model
    transformer_model_args = {
        "seq_input_dim": len(train_dataset.ts_cols),
        "seq_model_dim": 256,          # You can adjust this as needed
        "seq_num_heads": 8,            # Number of attention heads
        "seq_num_layers": 6,           # Number of Transformer layers
        "nonseq_input_dim": len(train_dataset.nonseq_cols),
        "mlp_hidden_dim": 512,
        "num_classes": 2,
        "dropout": 0.3,
        "max_len": 5000                 # Maximum sequence length
    }


    transformer_model = TransformerMixedModel(**transformer_model_args)
    logger.info(f"Transformer Model initialized with args: {transformer_model_args}")

    transformer_model.to(device)
    # Generate model name
    transformer_model_name = model_naming(**transformer_model_args)
    transformer_model_save_dir = os.path.join(PROJECT_PATH, 'model_data', 'transformer_mixed')
    os.makedirs(transformer_model_save_dir, exist_ok=True)
    transformer_model_save_path = os.path.join(transformer_model_save_dir, transformer_model_name)

    # Calculate class weights for imbalanced loss
    classes = np.unique(divergence_df_train['label'])
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=divergence_df_train['label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    logger.info(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(transformer_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Train the Transformer model
    logger.info("Starting training of the Transformer model...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=transformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=50,
        lr=1e-4,
        device=device,
        log_interval=10,
        save_path=transformer_model_save_path,
        patience=5
    )

    # Load the best Transformer model
    logger.info("Loading the best Transformer model for testing...")
    transformer_model.load_state_dict(torch.load(transformer_model_save_path, map_location=device))

    # Evaluate the Transformer model on the test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(transformer_model, test_loader, device)
    logger.info(f"Test Metrics: {test_metrics}")

    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1_score': test_metrics['f1_score'],
        'test_confusion_matrix': test_metrics['confusion_matrix']
    }
    metrics_path = os.path.join(transformer_model_save_dir, 'metrics.pkl')
    pd.to_pickle(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")

    # Plot training and validation metrics
    logger.info("Plotting training results for the Transformer model...")
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, outdir=os.path.join(PROJECT_PATH, 'plots'))



def main_lstm():
    """
    Main function to train, evaluate, and test the LSTM model for divergence classification.
    """
    # Load data
    logger.info("Loading data...")
    df = pd.read_pickle(os.path.join(PROJECT_PATH, 'data', 'processed_data', 'training_data.pickle'))
    divergence_data = pd.read_pickle(os.path.join(PROJECT_PATH, 'data', 'processed_data', 'divergence_data.pickle'))

    # Filter 5-minute timeframe data
    price_df = df[df['timeframe'] == '5m'].copy()
    divergence_df = divergence_data['15m'].copy()  # Assuming '5m' key exists

    # Split divergence_df into train/val/test
    logger.info("Splitting data into train, validation, and test sets...")
    total_events = len(divergence_df)
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    train_count = int(total_events * train_ratio)
    val_count = int(total_events * val_ratio)
    test_count = total_events - train_count - val_count

    divergence_df_train = divergence_df.iloc[:train_count]
    divergence_df_val = divergence_df.iloc[train_count:train_count+val_count]
    divergence_df_test = divergence_df.iloc[train_count+val_count:]

    logger.info(f"Train events: {len(divergence_df_train)}, "
                f"Validation events: {len(divergence_df_val)}, "
                f"Test events: {len(divergence_df_test)}")

    # Prepare divergence_data for multiple timeframes (if applicable)
    # Assuming divergence_data contains multiple timeframes
    # Example: divergence_data = {'5m': ddf_5m, '15m': ddf_15m, '1h': ddf_1h, ...}
    # For simplicity, using only '5m' here
    divergence_data_subset = {'15m': divergence_df_train}

    # Initialize Dataset
    logger.info("Initializing datasets...")
    train_dataset = LSTMDivergenceDataset(divergence_df=divergence_df_train, 
                                         price_df=price_df, 
                                         divergence_data=divergence_data, 
                                         seq_length=288)  # 288 * 5min = 24 hours
    # Use the same scaler for validation and test
    scaler = train_dataset.scaler
    val_dataset = LSTMDivergenceDataset(divergence_df=divergence_df_val, 
                                       price_df=price_df, 
                                       divergence_data=divergence_data, 
                                       seq_length=288, 
                                       scaler=scaler)
    
    test_dataset = LSTMDivergenceDataset(divergence_df=divergence_df_test, 
                                        price_df=price_df, 
                                        divergence_data=divergence_data, 
                                        seq_length=288, 
                                        scaler=scaler)

    # Initialize DataLoaders
    logger.info("Creating DataLoaders...")
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize Model
    model_args = {
        "seq_input_dim": len(train_dataset.ts_cols),
        "seq_hidden_dim": 128,
        "seq_num_layers": 3,
        "nonseq_input_dim": len(train_dataset.nonseq_cols),
        "mlp_hidden_dim": 256,
        "num_classes": 2,
        "dropout": 0.3
    }
    model = MixedLSTMModel(**model_args)
    logger.info(f"Model initialized with args: {model_args}")

    # Generate model name
    model_name = model_naming(**model_args)
    model_save_path = os.path.join(PROJECT_PATH, 'model_data', 'mixed_lstm', model_name)

    # Train the model
    logger.info("Starting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=1e-3,
        device=device,
        log_interval=10,
        save_path=model_save_path
    )

    # Load the best model
    logger.info("Loading the best model for testing...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # Evaluate on test set
    logger.info("Evaluating on test set...")
        # After evaluating on the test set
    test_metrics = evaluate_model(model, test_loader, device)
    logger.info(f"Test Metrics: {test_metrics}")

    metrics_path = os.path.join(PROJECT_PATH, 'model_data', 'mixed_lstm', 'metrics.pkl')
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy']
    }
    # Save the extended metrics
    metrics['test_precision'] = test_metrics['precision']
    metrics['test_recall'] = test_metrics['recall']
    metrics['test_f1_score'] = test_metrics['f1_score']
    metrics['test_confusion_matrix'] = test_metrics['confusion_matrix']

    pd.to_pickle(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")

    # Plot training and validation metrics
    logger.info("Plotting training results...")
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, outdir=os.path.join(PROJECT_PATH, 'plots'))


    
    
if __name__ == "__main__":
    main_transformer()