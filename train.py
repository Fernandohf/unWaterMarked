"""
Trainer for the model.
"""
import os
import datetime
import glob
import numpy as np
import torch
import torch.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer():
    """
    Trainer to train the model.
    """

    def __init__(self, train_dl, test_dl, model, optimizer,
                 scheduler, criterion, device='cuda'):
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, n_epochs=1, init_val_loss=np.Inf,
              print_every=300, init_epoch=0):
        """
        Function to train the model

        Args:
            batch_size: Batch size used
            train_dataloader: Dataloader used
            test_dataloader: Dataloader used
            n_epochs: Max number of epochs.
            init_val_loss: Initial validation loss.
            print_very: print this number of batches.
            train_device: device used for training.
        """
        # Losses array
        losses = {"train": [], "valid": []}
        valid_loss_min = init_val_loss
        # Progress bar
        pbar_epochs = tqdm(range(init_epoch, n_epochs + init_epoch),
                           total=n_epochs + init_epoch, ncols=500)
        for epoch in pbar_epochs:
            # keep track of training and validation loss
            train_loss = 0.0
            ###################
            # train the model #
            ###################
            self.model = self.model.train().to(self.device)
            total_train = len(self.train_dl.dataset)
            for c, (img, target) in enumerate(self.train_dl, 1):
                # Move tensors to GPU, if CUDA is available
                img, target = img.to(self.device), target.to(self.device)
                # Clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # Batch size
                bs = img.size(0)
                # forward pass
                pred = self.model(img)
                # calculate the batch loss
                # import ipdb; ipdb.set_trace() # debugging starts here
                # Fix dimension
                loss = self.criterion(pred, target)
                # backward pass: compute gradient of the loss
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update training loss
                train_loss += loss.item() * bs
                # show partial results
                if c % print_every == 0:
                    # print training statistics
                    pbar_epochs.write(
                        'Batch: {:3d}/{:3d} Training Loss: {:2.6f}'.format(
                            c,
                            len(self.train_dl),
                            train_loss / (bs * c)
                        ))

            # Validate model
            valid_loss = self.validate_model()
            losses["valid"].append(valid_loss)
            # calculate average losses
            train_loss = train_loss / total_train
            losses["train"].append(train_loss)
            # Save the model if validation loss has decreased
            save_model = False
            if valid_loss < valid_loss_min:
                # update the min_loss
                valid_loss_min = valid_loss
                # Saves the model
                save_model = True
            # Save results
            improve = 'better' if save_model else 'worse'
            self.save_checkpoint(epoch, losses, "Epoch_" + str(epoch) + '_' + improve)

            # print training/validation statistics
            output_str = ('Epoch: {:3d}/{:3d}' +
                          ' Training Loss: {:2.6f}' +
                          ' Validation Loss: {:2.6f}' +
                          ' Saving Model: {}')
            pbar_epochs.set_description(output_str.format(epoch, n_epochs, train_loss,
                                                          valid_loss, save_model))

            # Scheduler step
            self.scheduler.step(train_loss)

        # Return losses
        self.losses = losses

    def validate_model(self):
        """
        Validate the given model on test data.
        """
        ######################
        # validate the model #
        ######################
        # Initialize valid loss
        valid_loss = 0.0
        total_valid = len(self.test_dl.dataset)
        # Evaluation mode
        self.model = self.model.eval().to(self.device)
        with torch.no_grad():
            for img, target in self.test_dl:
                # move tensors to GPU if CUDA is available
                img, target = img.to(self.device), target.to(self.device)
                bs = img.size(0)
                # Check cache
                pred = self.model(img)
                # Calculate the batch loss
                loss = self.criterion(pred, target)
                # Update average validation loss
                valid_loss += loss.item() * bs
        # Valid loss
        valid_loss = valid_loss / total_valid
        return valid_loss

    def save_checkpoint(self, epoch, losses, file_name, directory="models"):
        """
        Saves the current model checkpoint

        Args:
            model: Model used.
            optimizer: Optimizer used.
            Scheduler: Scheduler used.
            epoch: Epoch number.
            losses: Dict with the losses.
            file_name: name of the saved file.
            directory: directory to save models.

        """
        # Append current datetime
        file_name += '_{date:%Y-%m-%d_%H-%M-%S}.model'.format(date=datetime.datetime.now())
        directory_name = os.path.join(directory, file_name)
        # Saves the model
        checkpoint = {"model_state_dict": self.model.state_dict(),
                      "optim_state_dict": self.optimizer.state_dict(),
                      "scheduler_state_dict": self.scheduler.state_dict(),
                      "epoch": epoch,
                      "train_loss": losses["train"],
                      "valid_loss": losses["valid"]}
        # Created directory
        torch.save(checkpoint, directory_name)


def find_best_model(path='models/*.model'):
    """
    Find the best model previously trained.
    """
    best = {'path': '', 'epoch': -1}
    for model_name in glob.glob(path):
        if 'better' in model_name:
            epoch = int(model_name.split('_')[1])
            if epoch > best['epoch']:
                best['path'] = model_name
                best['epoch'] = epoch
    print(f"Best model found: {best['path']}")
    return best


def load_checkpoint(file_path, model, optimizer, scheduler, location='cpu'):
    """
    Load all info from last model.

    Args:
        file_path: Relatice/full path to file.
        model: model to load weights from
        optimizer: optimizer to load parameters from.
        scheduler: to load from.
        location: Where to load the model.

    Return:
        Dict with all the weight loaded
    """
    # Loads the model
    checkpoint = torch.load(file_path, map_location=location)

    # Load in given objects
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    losses = {}
    losses["train"] = checkpoint["train_loss"]
    losses["valid"] = checkpoint["valid_loss"]

    return (model, optimizer, scheduler, losses)

if __name__ == "__main__":
    # Train Test Split
    cls_test = ['bicycle', 'bus', 'car', 'motorbike']
    dataset = VOCDetectionCustom(classes=cls_test)
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_len,
                                                                test_len])
    # PARAMETERS
    LR = 0.0256
    LR_FACTOR = .5
    PATIENCE = 5
    INIT_EPOCH = 0
    EPOCHS = 180
    BATCH_SIZE = 16
    CONTINUE = False
    train_dl = DataLoader(train_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

    test_dl = DataLoader(test_ds,
                         batch_size=BATCH_SIZE,
                         shuffle=True)
    # check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyYOLO(dataset.ANCHORS, len(dataset.classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer,
                                  'min',
                                  factor=LR_FACTOR,
                                  patience=PATIENCE)
    if CONTINUE:
        best_model = find_best_model()
        INIT_EPOCH = best_model['epoch']
        model, optimizer, scheduler, losses = load_checkpoint(best_model['path'], model, optimizer, scheduler, 'cuda')

    t = Trainer(train_dl, test_dl, model, optimizer, scheduler)
    t.train(n_epochs=EPOCHS, print_every=30, init_epoch=INIT_EPOCH)
