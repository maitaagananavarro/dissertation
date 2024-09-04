import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from emulator.src.core.evaluation import evaluate_preds

def calculate_confidence_thresholds(model, data_module, log):
    log.info("Calculating confidence thresholds")
    val_dataloaders = data_module.val_dataloader()
    confidences_dict = {}

    model.eval()
    with torch.no_grad():
        for dataloader in val_dataloaders:
            for batch in dataloader:
                inputs, targets = batch
                outputs_dict = model.predict(inputs, idx=None)
                for var_id in outputs_dict:
                    outputs = outputs_dict[var_id]
                    confidences = get_confidences(outputs)
                    if var_id not in confidences_dict:
                        confidences_dict[var_id] = []
                    confidences_dict[var_id].extend(confidences.cpu().numpy().flatten())

    thresholds_dict = {}
    for var_id, confidences in confidences_dict.items():
        threshold = np.percentile(confidences, 2.5)
        thresholds_dict[var_id] = threshold
        log.info(f"2.5th percentile confidence threshold for var_id {var_id}: {threshold:.4f}")

    return thresholds_dict

def evaluate_confidence_points(model, data_module, log, thresholds_dict, below_threshold=True):
    log.info("Evaluating points based on confidence threshold")
    test_dataloaders = data_module.test_dataloader()
    confidence_points = {var_id: {"inputs": [], "targets": []} for var_id in thresholds_dict}

    model.eval()
    with torch.no_grad():
        for dataloader in test_dataloaders:
            for batch in dataloader:
                inputs, targets = batch
                outputs_dict = model.predict(inputs, idx=None)
                split_targets = model.output_postprocesser.split_vector_by_variable(targets)

                for var_id in outputs_dict:
                    if var_id in thresholds_dict:
                        outputs = outputs_dict[var_id]
                        confidences = get_confidences(outputs)
                        threshold = thresholds_dict[var_id]

                        mask = confidences < threshold if below_threshold else confidences >= threshold

                        for i in range(inputs.size(0)):
                            if mask[i].any():
                                confidence_points[var_id]["inputs"].append(inputs[i])
                                if split_targets and var_id in split_targets:
                                    confidence_points[var_id]["targets"].append(split_targets[var_id][i])

    for var_id, data in confidence_points.items():
        num_points = len(data["inputs"])
        threshold_type = "below" if below_threshold else "above"
        log.info(f"Number of points {threshold_type} the 2.5th percentile threshold for var_id {var_id}: {num_points}")

    return confidence_points

def prepare_confidence_data(confidence_points):
    inputs_list = []
    targets_list = []

    for var_id in confidence_points:
        inputs_list.extend(confidence_points[var_id]["inputs"])
        targets_list.extend(confidence_points[var_id]["targets"])

    inputs_tensor = torch.stack(inputs_list) if inputs_list else torch.empty(0)
    targets_tensor = torch.stack(targets_list) if targets_list else torch.empty(0)

    dataset = TensorDataset(inputs_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataloader

def evaluate_on_confidence_points(model, dataloader, log):
    model.eval()
    val_step_outputs = {var_id: {"targets": [], "preds": []} for var_id in model._out_var_ids}

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            outputs_dict = model.predict(inputs, idx=None)

            for var_id in outputs_dict:
                outputs = outputs_dict[var_id]
                preds = outputs
                val_step_outputs[var_id]["targets"].append(targets)
                val_step_outputs[var_id]["preds"].append(preds)

    stats_per_var_id = {}
    for var_id in val_step_outputs:
        targets = torch.cat(val_step_outputs[var_id]["targets"], dim=0).cpu().numpy()
        preds = torch.cat(val_step_outputs[var_id]["preds"], dim=0).cpu().numpy()

        if targets.shape != preds.shape:
            raise ValueError(f"Dimensions of targets and preds do not match after adjustment for var_id {var_id}")

        stats = evaluate_preds(targets, preds)
        stats_per_var_id[var_id] = stats
        log.info(f"Evaluation metrics for var_id {var_id}: {stats}")

    return stats_per_var_id

def get_confidences(outputs):
    probabilities = F.softmax(outputs, dim=1)
    confidences, _ = torch.max(probabilities, dim=1)
    return confidences
