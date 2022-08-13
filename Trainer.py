import torch

class Trainer:
    
    def __init__(self, data_loader, network, device, learnrate, momentum):
        self.network = network
        self._dataloader = data_loader
        self._device = device
        self._learnrate = learnrate
        self._momentum = momentum
        self._optimizer = torch.optim.SGD(network.classifier.parameters(), lr=self._learnrate, momentum=self._momentum)
        self._output = torch.nn.LogSoftmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.training_losses = []
    
    
    def train(self):
        
        self.network.train()
        running_loss = 0
        
        for images, labels in self._dataloader:

            # move training inputs to computing device
            images, labels = images.to(self._device), labels.to(self._device)

            # zero the gradients
            self._optimizer.zero_grad()

            # forward propagate
            logps = self._output(self.network.forward(images))
            # measure training loss
            training_loss = self.criterion(logps, labels)
            running_loss += training_loss.item()
            # back propagate losses
            training_loss.backward()
            # update classifier weights
            self._optimizer.step()
        
        # record training losses for this epoch
        self.training_losses.append(running_loss/len(self._dataloader))