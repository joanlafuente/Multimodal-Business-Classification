from tqdm.auto import tqdm
import torch
import wandb

def train(model, train_loader, val_loader, criterion, optimizer, config):
    model.train()
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    best_val_loss = float('inf')
    best_val_acc = 0
    counter = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) # An squeduler that reduces the learning rate if the loss doesn't improve
    for epoch in tqdm(range(config.epochs)):
        for labels, img, text, text_mask in train_loader:
            loss = train_batch(img, text, text_mask, labels, model, optimizer, criterion)
            example_ct +=  len(labels)
            batch_ct += 1

            # Report the metrics of a batch every 25 batches
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
        
        # Evaluate the epoch results in the validation set
        val_loss, val_acc = val(model, val_loader, criterion, epoch)
        counter += 1

        # save the model if the validation loss is the lowest until now
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            counter = 0

        if config.patience < counter: # Early stopping criterium, if the validation loss doesn't improve for config.patience epochs, stop training
            break
        scheduler.step(val_loss) # We give the validation loss to the scheduler so it can reduce the learning rate if the validation loss doesn't improve

    model.load_state_dict(torch.load("best_model.pt")) # Load the best model, the one with a lower loss on the validation set
    return model

        


def train_batch(img, text, text_mask, labels, model, optimizer, criterion, device="cuda"):
    img, text, text_mask, labels = img.to(device), text.to(device), text_mask.to(device), labels.to(device)
    
    # Forward pass 
    outputs = model(img, text, text_mask)
    loss = criterion(outputs, labels)
    
    # Backward pass 
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss.item()


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

def val(model, val_loader, criterion, epoch, device="cuda"):
    # Computes the accuracy and loss in the validation set
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        losses = 0
        for labels, img, text, text_mask in val_loader:
            img, labels, text, text_mask = img.to(device), labels.to(device), text.to(device), text_mask.to(device)
            outputs = model(img, text, text_mask)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            losses += loss.item()

        losses /= len(val_loader)
        val_log(losses, correct / total, epoch)
        print(f"Accuracy of the model on the {total} " +
              f"val images: {correct / total:%}")
        return losses, correct / total

def val_log(loss, accuracy, epoch):
    # Logs the results of the validation set in weights and biases
    wandb.log({"epoch": epoch, "val_loss": loss, "val_acc":accuracy})