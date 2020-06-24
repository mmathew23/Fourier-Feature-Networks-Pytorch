import torch, fire
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from data import PositionDataset
from model import FourierNet
import torchvision


#Inputs
#fourier or not
#num layers
#num units
#image (or images)
#device id
#epochs
#batch_size

def train(
        image,
        num_layers=4,
        num_units=256,
        batch_size=4,
        learning_rate=1e-3,
        epochs=250,
        device_id=0,
        num_workers=8
        ):
    device = "cpu"
    if torch.cuda.is_available():
        device = f'cuda:{device_id}'

    #create dataset/loader
    dataset = PositionDataset(image)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    #instantiate model
    model = FourierNet(num_layers=num_layers, num_units=num_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            x, y = batch
            y = y.cuda(device)
            optimizer.zero_grad()
            y_hat = model(x.cuda(device))
            loss = mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            if epoch%10 == 0 and i == 0:
                print(f'Epoch: {epoch}. Loss: {loss}')
                torchvision.utils.save_image(y_hat, f'test_{epoch}.jpg')


if __name__ == '__main__':
  fire.Fire(train)
