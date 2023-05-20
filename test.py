import wandb
import torch

def test(model, test_loader, device="cuda", save:bool= True):
    # Run the model on some test examples
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for labels, img, text in test_loader:
            img, labels, text = img.to(device), labels.to(device), text.to(device)
            outputs = model(img, text)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    if save:
        # Save the model in the exchangeable ONNX format
        torch.onnx.export(model,  # model being run
                          (img, text),  # model input (or a tuple for multiple inputs)
                          "model.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        wandb.save("model.onnx")