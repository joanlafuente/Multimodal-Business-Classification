from tqdm.auto import tqdm
import torch
import wandb

def train(model, train_loader, val_loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    best_val_loss = float('inf')
    counter = 0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.1)    
    for epoch in tqdm(range(config.epochs)):
        for label, img, text in train_loader:
            loss = train_batch(img, text, label, model, optimizer, criterion)
            example_ct +=  len(label)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
        # Evaluate the epoch results oi the validation set
        val_loss = val(model, val_loader, criterion, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1 

        if config.patience < counter:
            break
        scheduler.step()
    model.load_state_dict(torch.load("best_model.pt"))
    return model

        


def train_batch(img, text, labels, model, optimizer, criterion, device="cuda"):
    img, text, labels = img.to(device), text.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(img, text)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

def val(model, val_loader, criterion, epoch, device="cuda"):
    with torch.no_grad():
        correct, total = 0, 0
        for labels, img, text in val_loader:
            img, labels, text = img.to(device), labels.to(device), text.to(device)
            outputs = model(img, text)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_log(loss, correct / total, epoch)
        print(f"Accuracy of the model on the {total} " +
              f"val images: {correct / total:%}")
        return loss.item()

def val_log(loss, accuracy, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "val_loss": loss, "val_acc":accuracy})