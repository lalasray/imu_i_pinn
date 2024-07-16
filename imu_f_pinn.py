import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output(x)
        return x

a_x_true, b_x_true = -2.0, 20.0

def generate_data(n_samples):
    t = torch.linspace(0, 100, n_samples).view(-1, 1)
    x = torch.sin(t)  # Example function for x(t)
    g_x = torch.cos(t)  # Example function for g_x(t)
    a_x_obs = a_x_true * x + b_x_true * g_x  # Observed acceleration x
    return t, x, g_x, a_x_obs

def loss_fn(model, t, g_x, a_x_obs):
    def predict(t):
        return model(t)

    # Compute second derivatives
    def second_derivative(t):
        t.requires_grad_(True)
        x_pred = predict(t)
        x_t1 = torch.autograd.grad(x_pred.sum(), t, create_graph=True)[0]
        x_t2 = torch.autograd.grad(x_t1.sum(), t, create_graph=True)[0]
        return x_t2

    x_t2 = second_derivative(t)

    a_x_pred = a_x_true * x_t2 + b_x_true * g_x
    data_loss = torch.mean((a_x_pred - a_x_obs) ** 2)
    residual_loss_x = torch.mean((x_t2 + g_x - a_x_pred) ** 2)

    return data_loss + residual_loss_x

def train_model(t, g_x, a_x_obs, n_epochs=5000):
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, t, g_x, a_x_obs)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        loss_history.append(loss.item())

    return model, loss_history

t, x, g_x, a_x_obs = generate_data(1000)
t.requires_grad_(True)
model, loss_history = train_model(t, g_x, a_x_obs)
def compute_second_derivative(model, t):
    t = t.clone().detach().requires_grad_(True)
    x_pred = model(t)
    x_t1 = torch.autograd.grad(x_pred.sum(), t, create_graph=True)[0]
    x_t2 = torch.autograd.grad(x_t1.sum(), t, create_graph=True)[0]
    return x_t2

t_pred = t.clone().detach().requires_grad_(True)
x_t2_pred = compute_second_derivative(model, t_pred)
a_x_pred = a_x_true * x_t2_pred + b_x_true * g_x

print(model.output.weight)
print(model.output.bias)
plt.figure(figsize=(12, 6))
plt.plot(t.detach().numpy(), a_x_obs.detach().numpy(), label='Observed Acceleration')
plt.plot(t.detach().numpy(), a_x_pred.detach().numpy(), label='Predicted Acceleration', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('Real vs Estimated Acceleration')
plt.legend()
plt.show()