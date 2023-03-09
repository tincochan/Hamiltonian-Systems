import torch
from tqdm import tqdm


def rk_step_closure(ys,neural_net,integrator,optimizer,dt):
    def closure():
        y1,y2 = ys[0:-1],ys[1:]
        optimizer.zero_grad()
        loss = torch.mean((y2 - integrator.integrator(y1,y2,neural_net,dt))**2)
        loss.backward(retain_graph=True)
        loss_value.append(loss.item()/(dt**2))
        return loss
    return closure


def train_on_seq(neural_net,optimizer,ys_train,dt,integrator,epochs):

    n = ys_train.shape[0]

    global loss_value
    with tqdm(range(epochs), unit="epochs") as tepoch:
        loss_value = []
        for epoch in tepoch:

            closure = rk_step_closure(ys_train,neural_net,integrator,optimizer,dt)

            optimizer.step(closure)

            tepoch.set_postfix(loss=loss_value[-1])