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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    for epoch in tqdm(range(config.epochs)):
        for label, img, text, text_mask in train_loader:
            loss = train_batch(img, text, text_mask, label, model, optimizer, criterion)
            example_ct +=  len(label)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
        # Evaluate the epoch results oi the validation set
        val_loss, val_acc = val(model, val_loader, criterion, epoch)
        if val_loss < best_val_loss:
            counter = 0
        else:
            counter += 1 

        # save the model if the validation accuracy is the best we've seen so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_transformer_keras.pt")

        if config.patience < counter:
            break
        scheduler.step(val_loss)
    model.load_state_dict(torch.load("best_model_transformer_keras.pt"))
    return model

        


def train_batch(img, text, text_mask, labels, model, optimizer, criterion, device="cuda"):
    img, text, text_mask, labels = img.to(device), text.to(device), text_mask.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(img, text, text_mask)
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
    model.train()
    with torch.no_grad():
        correct, total = 0, 0
        for labels, img, text, text_mask in val_loader:
            img, labels, text, text_mask = img.to(device), labels.to(device), text.to(device), text_mask.to(device)
            outputs = model(img, text, text_mask)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_log(loss, correct / total, epoch)
        print(f"Accuracy of the model on the {total} " +
              f"val images: {correct / total:%}")
        return loss.item(), correct / total

def val_log(loss, accuracy, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "val_loss": loss, "val_acc":accuracy})