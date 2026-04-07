import numpy as np
from DataCompolation.NMRFeatureGenerator import get_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NMRModel import NMR1DCNN

# Load the data
x, Y, T, Names, features = get_data()

# Remove duplicates from Y, T, Names, and features based on Names
Y = np.array(Y)
T = np.array(T)
Names = np.array(Names)
features = np.array(features)
idx = np.unique(Names, return_index=True)[1]
Y = Y[idx]
T = T[idx]
Names = Names[idx]
features = features[idx]



# Remove names list
List = ['2-Methylcyclohexanol']
idx = np.where(Names == List[0])
Y = np.delete(Y, idx[0], axis=0)
T = np.delete(T, idx[0], axis=0)
Names = np.delete(Names, idx[0], axis=0)
features = np.delete(features, idx[0], axis=0)

#save names 
np.savetxt('NamesNMR.csv',Names,delimiter=',',fmt='%s')

print(x.shape)

A = np.ones((len(Y), 1))
B = np.ones((len(Y), 1))
for i in range(len(Y)):
    y = Y[i]
    y = np.array(y)
    y = np.log(y)
    t = T[i]
    t = np.array(t)

    y = y[t > 250]
    t = t[t > 250]
    y = y[t < 600]
    t = t[t < 600]
    name = Names[i]
    
    # Fit A + B*t
    a = np.ones((len(t), 1))
    b = t.reshape(-1, 1)
    X = np.concatenate((a, 1 / b), axis=1)
    linear = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X.dot(linear[0])
    r2 = r2_score(y, y_hat)
    print(f'{name} R2: {r2}')
    A[i] = linear[0][0]
    B[i] = linear[0][1]

Y = np.hstack((A, B))
print(Y.shape)

# Standardize the data
scaler = StandardScaler()
Y = scaler.fit_transform(Y)

# Define the network
model = NMR1DCNN(input_length=6554)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Define the loss function
criterion = nn.MSELoss()

# Define the number of epochs
num_epochs = 500

# Convert the data to PyTorch tensors
x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # Add a channel dimension
y_tensor = torch.tensor(Y, dtype=torch.float32)

# Train-test split
idx = np.arange(x_tensor.shape[0])
np.random.shuffle(idx)
idx_train, idx_val = idx[:int(0.99*len(idx))], idx[int(0.99*len(idx)):]
x_tensor_train, x_tensor_val = x_tensor[idx_train], x_tensor[idx_val]
y_tensor_train, y_tensor_val = y_tensor[idx_train], y_tensor[idx_val]
names_train, names_val = np.array(Names)[idx_train], np.array(Names)[idx_val]
t_train, t_val = np.array(T)[idx_train], np.array(T)[idx_val]

# Track losses
train_losses = []
val_losses = []

for epoch in range(num_epochs):

    model.train()
    # Forward pass
    outputs = model(x_tensor_train)

    # Compute the training loss
    train_loss = criterion(outputs, y_tensor_train)

    # Zero the gradients
    optimizer.zero_grad()

    # Backward pass
    train_loss.backward()

    # Update the weights
    optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_tensor_val)
        val_loss = criterion(val_outputs, y_tensor_val)

    # Store the losses
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    # Print the losses
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

# Plot the training and validation loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(x_tensor_val)

# Calculate the R-squared value
r2 = r2_score(y_tensor_val.numpy(), y_pred.numpy())
print(f'R-squared: {r2}')

#unscale the data
y_pred = scaler.inverse_transform(y_pred.numpy())
y_tensor_val = scaler.inverse_transform(y_tensor_val.numpy())


# Plot the true vs predicted results
plt.figure()
for i in range(y_pred.shape[0]):
    y_val_temp = y_tensor_val[i][0] + y_tensor_val[i][1] * (1 / t_val[i])
    y_pred_temp = y_pred[i][0] + y_pred[i][1] * (1 / t_val[i])
    plt.scatter(y_val_temp, y_pred_temp, label=names_val[i])

plt.plot([-10, 10], [-10, 10], 'r--')
plt.xlim(y_pred.min(), y_pred.max())
plt.ylim(y_pred.min(), y_pred.max())
plt.legend()
plt.xlabel('True')  
plt.ylabel('Predicted')
plt.title('True vs. Predicted')
plt.show()

# Save the model
torch.save(model.state_dict(), 'model.pth')

