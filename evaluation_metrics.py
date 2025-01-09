from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch

def calculate_metrics(model, dataloader, device, average='macro'):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get the predicted class
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Calculate precision, recall, and F1 score with macro average
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=average, zero_division=0)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Prepare the metrics as a dictionary
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }
    
    return metrics
