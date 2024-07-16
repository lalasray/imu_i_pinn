import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

n_samples = 5000
alpha_x_true, beta_x_true = 13.0, 0.7
g_x_true = 10.0

def generate_data(n_samples, g_x_true=10.0, alpha_x_true=13.0, beta_x_true=0.7):
    t = torch.linspace(0, 100, n_samples).view(-1, 1)
    t.requires_grad_(True) 
    x = torch.sin(t)
    x.requires_grad_(True) 
    g_x = torch.full_like(t, g_x_true)

    
    alpha_x_times_x = alpha_x_true * x
    d_alpha_x_dt = torch.autograd.grad(outputs=alpha_x_times_x, inputs=t,
                                      grad_outputs=torch.ones_like(alpha_x_times_x),
                                      create_graph=True, retain_graph=True)[0]

    
    d2_alpha_x_dt2 = torch.autograd.grad(outputs=d_alpha_x_dt, inputs=t,
                                        grad_outputs=torch.ones_like(d_alpha_x_dt),
                                        create_graph=True)[0]

    
    a_x_obs = d2_alpha_x_dt2 + beta_x_true * g_x

    return t, x, g_x, a_x_obs


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_a = nn.Linear(128, 1)  # Predicted acceleration a_x
        self.fc_alpha = nn.Linear(128, 1)  # Estimated alpha_x
        self.fc_beta = nn.Linear(128, 1)  # Estimated beta_x

    def forward(self, x):
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
        x_pred = predict(t)
        grad_x = torch.autograd.grad(x_pred, t, create_graph=True, grad_outputs=torch.ones_like(x_pred))[0]
        grad_x_t = torch.autograd.grad(grad_x, t, create_graph=True, grad_outputs=torch.ones_like(grad_x))[0]
        return grad_x_t

    x_t2 = second_derivative(t)

    
    a_x_pred, alpha_x_pred, beta_x_pred = model(t)
    data_loss = torch.mean((a_x_pred - a_x_obs) ** 2)
    residual_loss_x = torch.mean((alpha_x_pred * x_t2 + beta_x_pred * g_x - a_x_pred) ** 2)

    total_loss = data_loss + residual_loss_x

    return total_loss


def train(model, optimizer, epochs):
    t, x, g_x, a_x_obs = generate_data(n_samples)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, t, g_x, a_x_obs)
        loss.backward(retain_graph=True)  
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')


def evaluate(model, n_samples=5000):
    model.eval()
    with torch.no_grad():
        t_eval = torch.linspace(0, 100, n_samples).view(-1, 1)
        a_x_pred, alpha_x_pred, beta_x_pred = model(t_eval)
    return t_eval, a_x_pred, alpha_x_pred, beta_x_pred


model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, optimizer, epochs=5000)

t_eval, a_x_pred, alpha_x_pred, beta_x_pred = evaluate(model)

print(f'Estimated alpha_x: {alpha_x_pred.mean()}, Estimated beta_x: {beta_x_pred.mean()}')

plt.figure(figsize=(10, 6))
plt.plot(t_eval.numpy(), a_x_pred.numpy(), label='Predicted acceleration (a_x)')
plt.plot(t_eval.numpy(), alpha_x_true * torch.sin(t_eval).numpy() + beta_x_true * g_x_true, label='True acceleration (a_x)')
plt.title('PINN Prediction vs True Acceleration')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.show()
