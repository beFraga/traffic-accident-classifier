from cnn.dataset import TrafficDataManager, count_occupied_cell_classes, effective_number_weights
from cnn.model import AccidentClassifier

import torch
import sys
import os
import random
import time
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORKDIR = Path().absolute()
DATASET_PATH = WORKDIR / "dataset"

print("Work directory: %s" % WORKDIR)
print("Dataset directory: %s" % DATASET_PATH)

with open("./parameters.yaml", "r") as yf:
    parameters = yaml.load(yf, Loader=yaml.SafeLoader)

params = parameters["accident_classifier"]

SAVE_DIR = WORKDIR / "training"

CLASS_NAMES = ["No accident", "Minor", "Moderate", "Severe", "Totaled Vehicle"]


def resolve_class_weights():
    """Resolve the per-class CE weights for the loss, reading config by key from
    ``parameters.yaml`` (per the project convention -- never hardcoded in cnn/).

    ``class_weight_mode: manual`` uses the literal ``class_w`` vector; otherwise the
    weights are derived from real label frequencies at startup. Falls back to uniform
    when no labels are on disk (e.g. dataset/ not yet restored) so training/eval can
    still construct the loss.
    """
    mode = params.get("class_weight_mode", "effective_number")

    if mode == "manual":
        manual = params.get("class_w")
        if manual is None:
            raise ValueError("class_weight_mode: manual requires a class_w vector in parameters.yaml")
        return [float(w) for w in manual]

    counts = count_occupied_cell_classes(DATASET_PATH, S=params["S"], num_classes=params["num_classes"])
    if counts.sum() == 0:
        print("⚠ No label files found under dataset/labels — falling back to uniform class weights.")
        return [1.0] * params["num_classes"]

    weights = effective_number_weights(
        counts,
        beta=params.get("class_weight_beta", 0.999),
        max_ratio=params.get("class_weight_max_ratio", 10.0),
        class0_cap=params.get("no_accident_weight_cap"),
    )
    print("----- Class weights (effective-number, occupied-cell frequencies) -----")
    for name, c, w in zip(CLASS_NAMES, counts.tolist(), weights):
        print(f"  {name:<16} count={c:<7} weight={w:.3f}")
    return weights


def set_seed(seed):
    """Seed the training-process RNGs (head init, train-loader shuffle, photometric
    augmentation). The val/test split has its own fixed generator (seed 42 in
    TrafficDataManager), so the held-out eval set is unaffected -- this isolates
    training stochasticity for the variance check. The seed comes from $TAC_SEED if
    set, else the ``seed`` key in parameters.yaml."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_seed():
    return int(os.environ.get("TAC_SEED", params.get("seed", 42)))


def train():
    start_time = time.time()
    seed = resolve_seed()
    set_seed(seed)
    print(f"----- Random seed: {seed} -----")
    print("----- Starting AccidentClassifier Train -----")
    print("----- Generating Traffic Accidents Dataset -----")
    print(f"----- Running on device: {device} --------")
    dataset = TrafficDataManager(
        DATASET_PATH,
        batch_size=params['batch_size'],
        S=params["S"],
        num_classes=params["num_classes"],
    )

    print(f"Generated {len(dataset)} samples")
    params["class_w"] = resolve_class_weights()
    model = AccidentClassifier(SAVE_DIR, dataset, params, device=device, early_stopping=True, tp=1)
    model.train()

    print("Total training time (CNN):")
    print(time.time() - start_time)
    print("Loss total (CNN):")
    print(model.history["train_loss_total"][-1])
    plt.figure()
    plt.plot(model.history["train_loss_total"])
    plt.plot(model.history["validation_loss_total"])
    plt.title("Loss")
    plt.legend(["Train", "Validation"], loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Score")
    loss_curve_path = SAVE_DIR / "loss_curve.png"
    plt.savefig(loss_curve_path, dpi=110)
    plt.close()
    print(f"Saved loss curve → {loss_curve_path}")
    run(dataset=dataset)



def run(dataset=None):
    print("----- Starting AccidentClassifier Test -----")
    if dataset is None:
        print("----- Generating Traffic Accidents Dataset -----")
        dataset = TrafficDataManager(
            DATASET_PATH,
            batch_size=params['batch_size'],
            S=params["S"],
            num_classes=params["num_classes"],
        )

    print(f"Generated {len(dataset)} samples")
    params["class_w"] = resolve_class_weights()
    model = AccidentClassifier(SAVE_DIR, dataset, params, device=device, tp=1)
    model.load_network()

    print("----- Evaluating Model Performance on Test Set -----")
    model.net.eval()
    
    test_loss_numerics = {key: 0.0 for key in model.loss.key_names}
    count_loop = 0
    
    with torch.no_grad():
        for images, targets in model.test_dataset:
            count_loop += 1
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model.net(images)
            loss_dict = model.loss(predictions, targets)

            img = images[0].detach()
            pred = predictions[0].detach()

            visualize_grid_predictions(img, pred, conf_threshold=0.5)
            
            for key in model.loss.key_names:
                test_loss_numerics[key] += loss_dict[key].item()
                
    print("\n================ TEST RESULTS ================")
    for key in model.loss.key_names:
        avg_test_loss = test_loss_numerics[key] / count_loop
        print(f"Average Test {key.capitalize()} Loss: {avg_test_loss:.5f}")
    print("==============================================")
    plot_traffic_confusion_matrix(model.net, model.test_dataset, device, conf_threshold=0.4)


def plot_traffic_confusion_matrix(model, loader, device, conf_threshold=0.5):
    """
    Extracts predictions for all occupied grid cells and plots a 5x5 confusion matrix.
    """
    model.eval()
    
    class_names = ["No accident", "Minor", "Moderate", "Severe", "Totaled Vehicle"]
    
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            # Locate cells where an accident exists and where the model opens the prediction gate
            obj_probs = torch.sigmoid(outputs[..., 0])
            pred_object = (obj_probs > conf_threshold)
            true_object = (targets[..., 0] == 1.0)
            
            # Isolate True Positive cells where both matrices match on an object's existence
            match_mask = (pred_object == True) & (true_object == True)
            
            if match_mask.sum() > 0:
                raw_class_logits = outputs[match_mask][..., 1:1 + model.num_classes]
                
                # Apply Softmax to establish clean category distributions
                softmax_probs = F.softmax(raw_class_logits, dim=-1)
                
                pred_indices = torch.argmax(softmax_probs, dim=-1).cpu().numpy()
                true_indices = torch.argmax(targets[match_mask][..., 1:1 + model.num_classes], dim=-1).cpu().numpy()
                
                all_pred_labels.extend(pred_indices)
                all_true_labels.extend(true_indices)
                
    if len(all_true_labels) == 0:
        print("\n❌ Zero matching true positive classifications found. The network didn't successfully spot any objects to classify.")
        print("   Try lowering your 'conf_threshold' (e.g., to 0.3) to allow weaker predictions through for analysis.")
        return

    # --- ACCURACY & METRICS CALCULATIONS ---
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    # Calculate global overall accuracy matching the diagonal of the matrix
    overall_class_accuracy = np.mean(all_true_labels == all_pred_labels)
    
    # Generate an explicit per-class breakdown report
    report = classification_report(
        all_true_labels, 
        all_pred_labels, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    
    print("\n" + "="*55)
    print(f" NETWORK CLASSIFICATION ACCURACY REPORT")
    print("="*55)
    print(f"OVERALL CLASSIFICATION ACCURACY: {overall_class_accuracy * 100:.2f}%\n")
    print(f"{'Accident Type':<18} | {'Precision (Reliability)':<23} | {'Recall (Detection Rate)':<20}")
    print("-"*72)
    
    for name in class_names:
        precision = report[name]['precision'] * 100
        recall = report[name]['recall'] * 100
        print(f"{name:<18} | {precision:6.2f}%{'':<16} | {recall:6.2f}%")
    print("="*75)

    # --- PLOT CONFUSION MATRIX ---
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(len(class_names))))
    cm_percent = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    
    labels = np.asarray([f"{v1}\n({v2:.1%})" for v1, v2 in zip(cm.flatten(), cm_percent.flatten())]).reshape(cm.shape)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_percent, 
        annot=labels, 
        fmt='', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Prediction Concentration'}
    )
    
    plt.title(f'Accident Type Confusion Matrix (Accuracy: {overall_class_accuracy * 100:.1f}%)', fontsize=12, pad=15)
    plt.xlabel('Predicted Accident Category')
    plt.ylabel('Actual Accident Category (Ground Truth)')
    cm_path = SAVE_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=110)
    plt.close()
    print(f"Saved confusion matrix → {cm_path}")

def visualize_grid_predictions(image_tensor, network_output, conf_threshold=0.35):
    """
    Visualizes the 10 output channels over the image.
    
    Args:
        image_tensor (Tensor): Input image of shape (3, H, W)
        network_output (Tensor): Single network prediction of shape (7, 7, 10)
        conf_threshold (float): Threshold to clear out background noise
    """
    # 1. Convert PyTorch image tensor to a plottable NumPy array (H, W, 3)
    if isinstance(image_tensor, torch.Tensor):
        # Denormalize if your loader normalizes images (uncomment if needed)
        # image_tensor = image_tensor * 0.5 + 0.5 
        img = image_tensor.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
    else:
        img = image_tensor

    img_h, img_w, _ = img.shape
    S = network_output.shape[0] # Usually 7
    
    # 2. Define distinct color coding for the 5 classification categories
    class_names = ["No accident", "Minor", "Moderate", "Severe", "Totaled Vehicle"]
    class_colors = {
        0: "blue",      # No accident
        1: "green",     # Minor
        2: "orange",    # Moderate
        3: "red",       # Severe
        4: "purple"     # Totaled Vehicle
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    # Optional: Draw the 7x7 chess board grid lines so you can see cell boundaries
    for i in range(1, S):
        ax.axhline(y=(i * img_h / S), color='white', linestyle='--', alpha=0.3)
        ax.axvline(x=(i * img_w / S), color='white', linestyle='--', alpha=0.3)

    # 3. Step through all 49 grid cells (Windows)
    for row in range(S):
        for col in range(S):
            # Isolate the 10 channels for this specific grid cell
            cell_data = network_output[row, col]
            
            # --- CHANNEL 0: OBJECTNESS (Existence Gate) ---
            obj_prob = torch.sigmoid(cell_data[0]).item()
            
            # If the cell doesn't think an accident is here, skip it!
            if obj_prob < conf_threshold:
                continue
                
            # --- CHANNELS 1 to 5: CLASSIFICATION (Color Selector) ---
            class_logits = cell_data[1:6]
            class_probs = F.softmax(class_logits, dim=-1)
            pred_class_idx = torch.argmax(class_probs).item()
            
            box_color = class_colors[pred_class_idx]
            label_text = f"{class_names[pred_class_idx]} ({obj_prob:.2f})"

            # --- CHANNELS 6 to 9: REGRESSION (Box Geometry) ---
            # Your network has no activation on the final layer, 
            # so apply sigmoid to bound relative coordinates between 0 and 1
            bx = torch.sigmoid(cell_data[6]).item()
            by = torch.sigmoid(cell_data[7]).item()
            bw = torch.sigmoid(cell_data[8]).item()
            bh = torch.sigmoid(cell_data[9]).item()

            # Decode Cell-Relative (x,y) to global Image Coordinates
            # (col / S) finds the starting pixel of the cell, then we add the offset
            center_x = ((col + bx) / S) * img_w
            center_y = ((row + by) / S) * img_h
            
            # Decode Image-Relative (w,h) to absolute pixels
            box_w = bw * img_w
            box_h = bh * img_h

            # Convert Box Center coordinates to Top-Left Corner for Matplotlib
            xmin = center_x - (box_w / 2)
            ymin = center_y - (box_h / 2)

            # 4. Draw the Bounding Box
            rect = patches.Rectangle(
                (xmin, ymin), box_w, box_h, 
                linewidth=2.5, edgecolor=box_color, facecolor='none'
            )
            ax.add_patch(rect)

            # 5. Add text banner marking the cell responsible for this prediction
            ax.text(
                xmin, ymin - 5, label_text, 
                color="white", weight="bold",
                bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
            )
            
            # Draw a tiny dot at the exact center coordinates predicted by the cell
            ax.plot(center_x, center_y, marker='o', color=box_color, markersize=5)

    plt.title(f"Grid Detector Parsing Screen (Threshold: {conf_threshold})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    sys.exit(switch[sys.argv[1]]())
