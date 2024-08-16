###############################################################################
"""
    Course:     Bio-Inspired Intelligence and Learning for Aerospace- 
                Applications

    Code:       AE4350 
    Year:       2023/2024 Q5
    Topic:      Applying Deep Q-Learning to the Snake Game
 
    Student:    Lars van Pelt
    Stud. no:   5629632
    Email:      L.H.vanpelt@student.tudelft.nl    


    NOTE:       Run the agent.py file to run the project
    
"""
###############################################################################

# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    # Introduce the Linear Feed-forward Neural Q-Network with 1 hidden layer
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
     
    # Apply ReLu activation functions 
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # Save the model
    def save(self, directory, file_name = 'model.pth'):
        file_name = os.path.join(directory, file_name)
        torch.save(self.state_dict(), file_name)
    
    # Load a saved model
    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))
        
    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model      = model
        self.lr         = lr
        self.gamma      = gamma
        self.optimizer  = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion  = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, gameover):
        state       = torch.tensor(state, dtype = torch.float)
        next_state  = torch.tensor(next_state, dtype=torch.float)
        action      = torch.tensor(action, dtype = torch.long)
        reward      = torch.tensor(reward, dtype= torch.float)
        
        # Make sure that multiple different sizes can be handled by trainer
        if len(state.shape) == 1:
            # Shape (1, x)
            state       = torch.unsqueeze(state, 0)
            next_state  = torch.unsqueeze(next_state, 0)
            action      = torch.unsqueeze(action, 0)
            reward      = torch.unsqueeze(reward, 0)
            gameover    = (gameover, )
            
        # Apply Bellman equation
        # Predicted q values with current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(gameover)):
            Q_new = reward[idx]
            
            if not gameover[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                
            target[idx][torch.argmax(action).item()] = Q_new
            
        # new Q = R + gamma * max(next pred Q values)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()
        