import os
import logging

import torch
from MochaGCN.models.base_models import NCModel
import torch.nn as nn
from data import get_positive_negative_samples, construct_positive_negative_representations
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix

def hyperbolic_distance(x, y, eps=1e-5):
    # Ensure numerical stability
    sqnorm_x = torch.clamp(torch.sum(x ** 2, dim=-1), min=eps, max=1-eps)
    sqnorm_y = torch.clamp(torch.sum(y ** 2, dim=-1), min=eps, max=1-eps)

    inner_product = torch.sum(x * y, dim=-1)

    # Ensure the values inside the sqrt are non-negative
    sqnorm_prod = sqnorm_x * sqnorm_y
    inner_prod_sq = inner_product ** 2
    numerator = 2 * (sqnorm_prod - inner_prod_sq)

    # Ensure the values are in a valid range for acosh
    cosh_distance = 1 + numerator / ((1 - sqnorm_x) * (1 - sqnorm_y) + eps)
    cosh_distance = torch.clamp(cosh_distance, min=1 + eps)

    return torch.acosh(cosh_distance)

def criterion(batch_embeddings, positive_embeddings, negative_embeddings, margin=1.0):
    # Calculate the hyperbolic distance for positive and negative pairs
    positive_distance = hyperbolic_distance(batch_embeddings, positive_embeddings)
    negative_distance = hyperbolic_distance(batch_embeddings, negative_embeddings)

    # Ensure positive distance is less than negative distance by a margin
    loss = F.relu(positive_distance - negative_distance + margin)

    return loss.mean()  # Use mean to aggregate losses across the batch



def validate_MochaGCN(test_loader, model_modal):
    model_modal.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_modal in test_loader:
            # positive_indices, negative_indices = get_positive_negative_samples(batch_modal1)

            batch_modal_adj_dense = to_dense_adj(batch_modal.edge_index).squeeze(0)
            batch_modal_embeddings = model_modal.encode(batch_modal.x, batch_modal_adj_dense)
            batch_modal_embeddings = model_modal.decode(batch_modal_embeddings, batch_modal_adj_dense)
            # batch_modal1_positive_embeddings, batch_modal1_negative_embeddings = construct_positive_negative_representations(
            #     batch_modal1_embeddings, positive_indices, negative_indices
            # )
            # batch_modal1_loss = criterion(batch_modal1_positive_embeddings, batch_modal1_negative_embeddings, model_modal1.parameters())
            # batch_modal1_loss = triplet_loss(batch_modal1_embeddings, batch_modal1_positive_embeddings, batch_modal1_negative_embeddings)



            output = global_mean_pool(batch_modal_embeddings, batch_modal.batch)
            # output = global_mean_pool((batch_modal1_embeddings + batch_modal2_embeddings), batch_modal1.batch)
            output = F.sigmoid(output)

            labels = F.one_hot(batch_modal.y, num_classes=2).float()

            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(output, labels)

            # loss = batch_modal1_loss + batch_modal2_loss + NLloss
            # loss = Xloss

            total_loss += loss.item() * batch_modal.num_graphs

            pred = output.argmax(dim=1)
            correct += (pred == batch_modal.y).sum().item()
            total += batch_modal.num_graphs

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_modal.y.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = correct / total

    # Calculate SEN, SPE, and AUC
    confusion = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = confusion.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, accuracy, sensitivity, specificity, auc

def train_MochaGCN(train_loader, test_loader, args):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    accs = {}

    model_modal = NCModel(args)
    optimizer_modal = torch.optim.Adam(params=model_modal.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model_modal.train()

        total_loss = 0
        correct = 0
        total = 0
        flag = True

        for batch_modal in train_loader:
            optimizer_modal.zero_grad()
            positive_indices, negative_indices = get_positive_negative_samples(batch_modal)

            batch_modal_embeddings = model_modal.encode(batch_modal.x, batch_modal.dense_adj)

            visualization_embeddings_temp = batch_modal_embeddings.clone()

            batch_modal_embeddings = model_modal.decode(batch_modal_embeddings, batch_modal.dense_adj)
            batch_modal_embeddings_pooled = global_mean_pool(batch_modal_embeddings, batch_modal.batch)
            batch_modal_positive_embeddings, batch_modal_negative_embeddings = construct_positive_negative_representations(
                batch_modal_embeddings_pooled, positive_indices, negative_indices
            )
            batch_modal_loss = criterion(batch_modal_embeddings_pooled, batch_modal_positive_embeddings,
                                          batch_modal_negative_embeddings)
            output = global_mean_pool(batch_modal_embeddings, batch_modal.batch)
            output = F.sigmoid(output)

            labels = F.one_hot(batch_modal.y, num_classes=2).float()

            loss_fn = torch.nn.BCELoss()
            BCEloss = loss_fn(output, labels)

            loss = batch_modal_loss * 0.01 + BCEloss

            total_loss += loss.item() * batch_modal.num_graphs

            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                torch.nn.utils.clip_grad_norm_(model_modal.parameters(), max_norm)

            pred = output.argmax(dim=1)
            correct += (pred == batch_modal.y).sum().item()
            total += batch_modal.num_graphs

            loss.backward()
            optimizer_modal.step()


            if flag == True:
                visualization_embeddings = global_mean_pool(visualization_embeddings_temp, batch_modal.batch)
                train_y = batch_modal.y.clone()
                output_all = output.clone()
                flag = False
            else:
                visualization_embeddings = torch.cat(
                    (visualization_embeddings, global_mean_pool(visualization_embeddings_temp, batch_modal.batch)),
                    dim=0)
                train_y = torch.cat((train_y, batch_modal.y.clone()), dim=0)
                output_all = torch.cat((output_all, output.clone()), dim=0)
        accuracy = correct / total

        train_accuracy = accuracy
        train_loss = total_loss / len(train_loader)
        #validate test

        test_loss, test_accuracy, test_sensitivity, test_specificity, test_auc = validate_MochaGCN(test_loader, model_modal)
        log_msg = (
            f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, '
            f'Test Accuracy: {test_accuracy:.4f}, Test Sensitivity: {test_sensitivity:.4f}, '
            f'Test Specificity: {test_specificity:.4f}, Test AUC: {test_auc:.4f}')
        print(log_msg)

        if test_accuracy > 0.8:
            logging.info(log_msg)

        accs[epoch + 1] = (test_accuracy, test_sensitivity, test_specificity, test_auc)

    return model_modal