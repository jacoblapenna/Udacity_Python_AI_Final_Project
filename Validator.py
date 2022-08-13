import torch

class Validator:
    
    def __init__(self, data_loader, device):
        self._dataloader = data_loader
        self._device = device
        self._output = torch.nn.LogSoftmax(dim=1)
        self.validation_losses = []
        self.accuracies = []
        
    
    
    def validate(self, network, criterion):
        
        with torch.no_grad():
            
            network.eval()
            running_loss = 0
            accuracy = 0
            
            # run through each validation batches
            for images, labels in self._dataloader:

                # move validation inputs to computing device
                images, labels = images.to(self._device), labels.to(self._device)

                # forward propagate
                logps = self._output(network.forward(images))
                # measure validation loss
                validation_loss = criterion(logps, labels)
                running_loss += validation_loss.item()
                ## claculate accuracy
                # get the ouput class probabilities
                probabilities = torch.exp(logps)
                # get the classification with highest probability from each validation image
                top_prob, top_class = probabilities.topk(1, dim=1)
                # convert to a list of bools on matches
                top_class_matches = top_class == labels.view(*top_class.shape)
                # accuracy is percentage of validation images correctly classified
                accuracy += torch.mean(top_class_matches.type(torch.FloatTensor)).item() * 100.0
            
            self.validation_losses.append(running_loss/len(self._dataloader))
            self.accuracies.append(accuracy/len(self._dataloader))
                