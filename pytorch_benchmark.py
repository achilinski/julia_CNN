
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import psutil
import os
import sys

from pytorch_model import IMDbCNN

def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

def calculate_accuracy(model, dataloader, device):
    model.eval() 
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad(): 
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            predicted = (outputs > 0.5).float() 
            total_predictions += y_batch.size(0)
            correct_predictions += (predicted == y_batch).sum().item()
    return correct_predictions / total_predictions

if __name__ == '__main__':
    print("Starting PyTorch CNN Benchmark (Full 5 Epochs with Accuracy)...")
    print("="*50)

    
    EMBEDDING_DIM = 50
    VOCAB_SIZE = 12849
    MAX_LEN = 130
    KERNEL_WIDTH = 3
    CONV_OUT_CHANNELS = 8
    POOL_SIZE = 8
    DENSE_IN_FEATURES = ((MAX_LEN - KERNEL_WIDTH + 1) // POOL_SIZE) * CONV_OUT_CHANNELS
    DENSE_OUT_FEATURES = 1
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 5
    
    
    
    PYTORCH_PAD_IDX = 12848 
    print(f"Using PYTORCH_PAD_IDX = {PYTORCH_PAD_IDX}")

    
    print("Loading pre-converted NumPy data...")
    try:
        X_train_np = np.load("X_train_pytorch.npy")
        y_train_np = np.load("y_train_pytorch.npy")
        X_test_np = np.load("X_test_pytorch.npy")   
        y_test_np = np.load("y_test_pytorch.npy")     
        embeddings_np = np.load("embeddings_pytorch.npy")
    except FileNotFoundError as e:
        print(f"ERROR: NumPy data file not found: {e.filename}")
        print("Please generate these from your Julia data using save_data_for_pytorch.jl.")
        exit()

    print(f"Train data: X_train_np shape: {X_train_np.shape}, y_train_np shape: {y_train_np.shape}")
    print(f"Test data: X_test_np shape: {X_test_np.shape}, y_test_np shape: {y_test_np.shape}")
    print(f"Embeddings_np shape: {embeddings_np.shape}")

    
    for name, arr in [("X_train", X_train_np), ("X_test", X_test_np)]:
        if np.max(arr) >= VOCAB_SIZE or np.min(arr) < 0:
            print(f"ERROR: Indices in {name} out of range for VOCAB_SIZE={VOCAB_SIZE}.")
            print(f"Min index: {np.min(arr)}, Max index: {np.max(arr)}")
            exit()
    if PYTORCH_PAD_IDX is not None and (PYTORCH_PAD_IDX < 0 or PYTORCH_PAD_IDX >= VOCAB_SIZE):
        print(f"ERROR: PYTORCH_PAD_IDX={PYTORCH_PAD_IDX} is out of range for VOCAB_SIZE={VOCAB_SIZE}.")
        exit()

    X_train_tensor = torch.from_numpy(X_train_np).long()
    y_train_tensor = torch.from_numpy(y_train_np).float()
    if y_train_tensor.ndim == 1: y_train_tensor = y_train_tensor.unsqueeze(1)

    X_test_tensor = torch.from_numpy(X_test_np).long()     
    y_test_tensor = torch.from_numpy(y_test_np).float()       
    if y_test_tensor.ndim == 1: y_test_tensor = y_test_tensor.unsqueeze(1)
    
    pre_trained_embeddings_tensor = torch.from_numpy(embeddings_np).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    
    print("Setting up PyTorch model...")
    device = torch.device("cpu") 
    print(f"Using device: {device}")

    pytorch_cnn_model = IMDbCNN(VOCAB_SIZE, EMBEDDING_DIM, pre_trained_embeddings_tensor,
                                KERNEL_WIDTH, CONV_OUT_CHANNELS, POOL_SIZE,
                                DENSE_IN_FEATURES, DENSE_OUT_FEATURES, 
                                pad_idx=PYTORCH_PAD_IDX).to(device)
    
    optimizer = optim.Adam(pytorch_cnn_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    print("PyTorch model and optimizer created.")

    
    def train_step_pytorch(model, opt, loss_fn, x_batch, y_batch):
        model.train()
        opt.zero_grad()
        outputs = model(x_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        opt.step()
        return loss.item(), outputs 

    
    print(f"Starting PyTorch training ({NUM_EPOCHS} epochs)...")
    initial_ram = get_process_memory()
    print(f"Initial RAM: {initial_ram:.2f} MB")
    overall_start_time = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.perf_counter()
        total_epoch_loss = 0.0
        total_epoch_train_acc_preds = 0 
        total_epoch_train_samples = 0   
        num_batches_in_epoch = 0
        
        pytorch_cnn_model.train() 
        for x_b, y_b in train_dataloader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            
            optimizer.zero_grad()
            outputs = pytorch_cnn_model(x_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
            
            predicted_train = (outputs > 0.5).float()
            total_epoch_train_acc_preds += (predicted_train == y_b).sum().item()
            total_epoch_train_samples += y_b.size(0)
            
            num_batches_in_epoch += 1
        
        epoch_end_time = time.perf_counter()
        avg_epoch_loss = total_epoch_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        avg_epoch_train_acc = total_epoch_train_acc_preds / total_epoch_train_samples if total_epoch_train_samples > 0 else 0
        
        
        test_acc_epoch = calculate_accuracy(pytorch_cnn_model, test_dataloader, device)
        
        
        
        print(f"PyTorch - Epoch: {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_epoch_loss:.4f}, Train Acc: {avg_epoch_train_acc:.4f}, Test Acc: {test_acc_epoch:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s")
        sys.stdout.flush()

    overall_end_time = time.perf_counter()
    total_training_time = overall_end_time - overall_start_time
    final_ram = get_process_memory()

    print("="*50)
    print("PyTorch CNN Training Finished.")
    print(f"Total PyTorch training time for {NUM_EPOCHS} epochs: {total_training_time:.2f} seconds")
    print(f"RAM at start: {initial_ram:.2f} MB, RAM at end: {final_ram:.2f} MB")
    print("Note: For true peak RAM, monitor with OS tools during execution.")
    print("="*50)