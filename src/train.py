# %% [code] {"execution":{"iopub.status.busy":"2025-11-15T19:30:07.534605Z","iopub.execute_input":"2025-11-15T19:30:07.535373Z","iopub.status.idle":"2025-11-15T19:30:11.044416Z","shell.execute_reply.started":"2025-11-15T19:30:07.535344Z","shell.execute_reply":"2025-11-15T19:30:11.043823Z"},"jupyter":{"outputs_hidden":false}}


import numpy as np
import pandas as pd 
import chess
import chess.pgn
import torch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader, random_split

import pandas as pd
import modal

app = modal.App("chessTraining")


pgn_volume= modal.Volume.from_name ('chess-data', create_if_missing =True)

image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy", 
    "pandas",
    "chess"
)













########################################################################################
@app.function(gpu="L24", timeout=72000, image=image, volumes ={"/data": pgn_volume})
def train_model():
    import torch
    import torch.nn as nn
    import chess
    import chess.pgn
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader, random_split
    
    #get data
    pgn_path = "/data/games-2.5s.pgn"
    #set up gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"Using device: {device}") #check if gpu is used
    
    game_list_pgn=[]
    with open (pgn_path, encoding ='utf-8') as pgn_file:
    
        while True:
            game= chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_list_pgn.append(game)
            



    def board_to_tensor(board : chess.Board): 
        #
        piece_encode= {
            chess.PAWN: 0, 
            chess.KNIGHT: 1, 
            chess.BISHOP: 2, 
            chess.ROOK:3, 
            chess.QUEEN: 4,
            chess.KING: 5 
            }

        tensor=np.zeros ( (13, 8, 8), dtype=np.float32) 
        #creates a 12 x 8 x 8 tensor. the 8x8 represents the board, 
        #while the array of 12 is acts as a one-hot encoding for the pieces

        #iterates over all squares on a chessboard
        if board.turn ==chess.WHITE:
            tensor[12, : ,:] =1.0
        for square in chess.SQUARES:
            piece=board.piece_at(square)
            if piece:
                row = chess.square_rank(square)
                col= chess.square_file(square)
                if piece.color == chess.WHITE:
                    piece_value= piece_encode[piece.piece_type]
                    tensor[piece_value, row, col] =1.0
                
                else: 
                    piece_value= piece_encode[piece.piece_type] + 6
                    tensor[piece_value, row, col]=1.0
            
        return tensor

    positions=[]
    moves=[]
    for game in game_list_pgn:
    
        board=game.board()
        for move in game.mainline_moves():
            positions.append(board.fen())
            moves.append(move.uci())
            board.push(move)


    def encode_move(move_list):
        encoded_list =[]
        for move in move_list:
            start = chess.parse_square(move[:2]) 
            end = chess.parse_square(move[2:4])
            encoded_list.append(start * 64 + end)

        return encoded_list

    pos_tensors=[]
    for pos in positions:
        pos_tensors.append(board_to_tensor(chess.Board(pos)))
    
        #we use np.stack to create one big array from the list of pos_tensors which is much faster
    x = torch.from_numpy(np.stack(pos_tensors)).float()
    y=torch.LongTensor(encode_move(moves))

    class CnnModel (nn.Module):
        def __init__(self):
            super().__init__()  # Initialize parent class
            self.features = nn.Sequential(
                # Conv block 1
                nn.Conv2d(13, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(), #activation
            
                # Conv block 2
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),#activation
            
                # Conv block 3
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),#activation
                )
       
            self.classifier = nn.Sequential(
                nn.Flatten(),                      # Step 1: (256, 8, 8) → (16384,)
                nn.Linear(256 * 8 * 8, 1024),     # Step 2: 16384 → 1024 neurons
                nn.ReLU(),                         # Step 3: Activation
                nn.Dropout(0.3),                   # Step 4: Regularization (prevent overfitting)
                nn.Linear(1024, 4096)             # Step 5: 1024 → 4096 (OUTPUT LAYER)
                )
    
        def forward(self, x):
            x = self.features(x)      # Step 1: go through feature layer and detect features
            x = self.classifier(x)    # Step 2: Make prediction
            return x

    model=CnnModel()

    dataset = TensorDataset(x, y) ##this is more efficient that inputting tensors directly
    train_size= int(0.8* len(dataset))
    val_size=len(dataset)-train_size

    
    

    train_ds, val_ds = random_split(dataset, [train_size, val_size])


    batch_size = 48  # Process 64 positions at a time
    #we use DataLoader so we can train in batches using the gpu which is much faster
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) 
    #shuffle the training so it doesn't learn patterns based on order
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    

    
    model = CnnModel().to(device)

    #loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ##reduce learning rate when plateuing

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,      # Reduce LR by half when plateau detected
    patience=2,      # Wait 2 epochs before reducing
    min_lr=1e-6
    )
    #early stopping parameters
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None

    num_epochs = 20  # Train for 20 epochs (can be stopped early)

    print(f"\n{'='*70}")
    print(f"STARTING TRAINING - {num_epochs} epochs")
    print(f"{'='*70}\n")

    

            #TRAINING LOOP
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 70)
    
        # ========== TRAINING PHASE ==========
        model.train()  # Set model to training mode
        train_loss = 0
        train_correct = 0
        train_total = 0
    
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # Move batch to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
        
            # Zero gradients from previous iteration
            optimizer.zero_grad()
        
            # Forward pass: get predictions
            output = model(batch_x)
        
            # Calculate loss
            loss = criterion(output, batch_y)
        
            # Backward pass: calculate gradients
            loss.backward()
        
            # Update weights
            optimizer.step()
        
            # Track statistics
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == batch_y).sum().item()
            train_total += batch_y.size(0)
        
            # Print progress every 1000 batches
            if batch_idx % 1000 == 0:
                current_acc = 100 * train_correct / train_total
                print(f"  Batch [{batch_idx:5d}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {current_acc:.2f}%")
    
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        
        # ========== VALIDATION PHASE ==========
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        val_correct = 0
        val_total = 0
    
        with torch.no_grad():  # Don't calculate gradients (saves memory)
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
            
                # Forward pass
                output = model(batch_x)
            
                # Calculate loss
                val_loss += criterion(output, batch_y).item()
            
                # Calculate accuracy
                pred = output.argmax(dim=1)
                val_correct += (pred == batch_y).sum().item()
                val_total += batch_y.size(0)
    
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"    Learning Rate: {current_lr:.6f}")

        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), '/data/best_model.pth')
            print(f"    ✅ New best model! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    ⚠️ No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"{'='*70}")
            model.load_state_dict(best_model_state)
            break
        
    
        # Print epoch summary
        print(f"\n  Results:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"    Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")
        print()

    print(f"{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"{'='*70}")
    #save the model
    # Save the model to the volume (not temporary directory!)
    torch.save(model.state_dict(), '/data/chess_model_Kiran.pth')  # ← FIX THIS LINE
    print("Model saved to /data/chess_model_Kiran.pth")

    pgn_volume.commit()

    return "success!"




#############################################################################




@app.local_entrypoint()
def main():
    print("Starting training on Modal cloud...")
    result = train_model.remote()
    print(result)



