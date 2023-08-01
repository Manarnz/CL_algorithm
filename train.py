import torch
import torch.nn as nn
from torch.autograd import Variable
import mydataset
import cnn_model
import heat_kernel

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cnn_model.CNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader = mydataset.get_train_data_loader()
    for inputs, kernel in train_loader:
        
        optimizer.zero_grad()
        outputs = model(inputs)
        kernel = kernel[0].to(torch.float32)
        loss = criterion(outputs, kernel)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), "kernel_prediction_model.pth")
