import wandb
import torch
import pickle

def test(model, test_loader, config, device="cuda", save:bool= True):
    # Run the model on some test examples
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for labels, img, text,text_mask in test_loader:
            img, labels, text, text_mask = img.to(device), labels.to(device), text.to(device), text_mask.to(device)
            outputs = model(img, text, text_mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    if save:
        # Save the model in a pickle format, ONNX does not allow us to save the transformer module
        with open(config.name_model, "wb") as f:
            pickle.dump({"model_weights":model.state_dict(), "parameters":dict(config)}, f)    