import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

n_samples = 5000
alpha_x_true, beta_x_true = 13.0, 0.7
g_x_true = 10.0

# Data is not acceleration but rather a*x + b
def generate_data(n_samples):
    t = torch.linspace(0, 100, n_samples).view(-1, 1)
    x = torch.sin(t)
    g_x = torch.full_like(t, g_x_true)
    a_x_obs = alpha_x_true * x + beta_x_true * g_x
    return t, x, g_x, a_x_obs

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_a = nn.Linear(128, 1)  
        self.fc_alpha = nn.Linear(128, 1)  
        self.fc_beta = nn.Linear(128, 1)  

    def forward(self, x):
        #x = torch.sin(x)  # x is sinusoidal in the data generation
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        a_x = self.fc_a(x)
        alpha_x = self.fc_alpha(x)
        beta_x = self.fc_beta(x)
        return a_x, alpha_x, beta_x

def loss_fn(model, t, g_x, a_x_obs):
    def predict(t):
        return model(t)[0] 

    def second_derivative(t):
        t.requires_grad_(True)
        x_pred = predict(t)
        x_t1 = torch.autograd.grad(x_pred.sum(), t, create_graph=True)[0]
        x_t2 = torch.autograd.grad(x_t1.sum(), t, create_graph=True)[0]
        return x_t2

    x_t2 = second_derivative(t)

    a_x_pred, alpha_x_pred, beta_x_pred = model(t)
    data_loss = torch.mean((a_x_pred - a_x_obs) ** 2)
    residual_loss_x = torch.mean((alpha_x_pred * x_t2 + beta_x_pred * g_x - a_x_pred) ** 2)

    return data_loss + residual_loss_x

def train(model, optimizer, epochs):
    t, x, g_x, a_x_obs = generate_data(n_samples)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, t, g_x, a_x_obs)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, optimizer, epochs=5000)

model.eval()
with torch.no_grad():
    t_eval = torch.linspace(0, 100, n_samples).view(-1, 1)
    a_x_pred, alpha_x_pred, beta_x_pred = model(t_eval)

print(f'Estimated alpha_x: {alpha_x_pred.mean()}, Estimated beta_x: {beta_x_pred.mean()}')

plt.figure(figsize=(10, 6))
plt.plot(t_eval.numpy(), a_x_pred.numpy(), label='Predicted acceleration (a_x)')
plt.plot(t.detach().numpy(), alpha_x_true * x.detach().numpy() + beta_x_true * g_x_true, label='True acceleration (a_x)')
plt.title('PINN Prediction vs True Acceleration')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.show()
