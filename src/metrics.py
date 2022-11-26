import torch
from torchmetrics import functional as F


# check if all classes are represented in target
def has_all_classes_represented(target, num_classes, is_multilabel):
    if is_multilabel:
        for c in range(num_classes):
            if len(torch.unique(target[:, c])) < 2:
                return False
        return True
    else:
        return len(torch.unique(target)) == num_classes


def calc_metrics(preds, target, is_multilabel=False):
    num_classes = preds.shape[1]

    if is_multilabel and target.dtype != torch.int:
        target = target.to(dtype=torch.int)

    metrics = {
        'accuracy': F.accuracy(preds, target),
        'macro_accuracy': F.accuracy(preds, target, num_classes=num_classes, average='macro'),
        'weighted_accuracy': F.accuracy(preds, target, num_classes=num_classes, average='weighted'),

        'f1_score': F.f1(preds, target, num_classes=num_classes, average='micro'),
        'macro_f1_score': F.f1(preds, target, num_classes=num_classes, average='macro'),
        'weighted_f1_score': F.f1(preds, target, num_classes=num_classes, average='weighted'),

        'precision_recall': F.precision_recall(preds, target, num_classes=num_classes, average='micro'),
        'macro_precision_recall': F.precision_recall(preds, target, num_classes=num_classes, average='macro'),
        'weighted_precision_recall': F.precision_recall(preds, target, num_classes=num_classes, average='weighted'),

        'matthews_corrcoef': F.matthews_corrcoef(preds, target, num_classes=num_classes),
    }

    if has_all_classes_represented(target, num_classes, is_multilabel):
        metrics['auroc'] = F.auroc(preds, target, num_classes=num_classes)
        metrics['macro_auroc'] = F.auroc(preds, target, num_classes=num_classes, average='macro')
        metrics['weighted_auroc'] = F.auroc(preds, target, num_classes=num_classes, average='weighted')

    return metrics


def eval_vae_classifier(classifier, eval_device, test_loader):
    step_y_true = []
    step_y_score = []

    with torch.no_grad():
        classifier.eval()
        classifier.to(eval_device)

        for batch in test_loader:
            x, labels = batch
            if not isinstance(x, list):
                x = x.to(eval_device)
            else:
                x = [x_i.to(eval_device) for x_i in x]

            logits = classifier(x)
            ps = torch.exp(logits)

            step_y_true.append(labels)
            step_y_score.append(ps)

        epoch_y_true = torch.cat(step_y_true)
        epoch_y_score = torch.cat(step_y_score)
        y_pred = torch.argmax(epoch_y_score, dim=1)

    return epoch_y_true.cpu().numpy(), epoch_y_score.cpu().numpy(), y_pred.cpu().numpy()


def eval_multirep_classifier(classifier, eval_device, test_loader):
    step_y_true = []
    step_y_score = []

    with torch.no_grad():
        classifier.eval()
        classifier.to(eval_device)

        for batch in test_loader:
            x_smi, x_img, labels = batch
            if not isinstance(x_smi, list):
                x_smi = x_smi.to(eval_device)
            else:
                x_smi = [x_i.to(eval_device) for x_i in x_smi]

            if not isinstance(x_img, list):
                x_img = x_img.to(eval_device)
            else:
                x_img = [x_i.to(eval_device) for x_i in x_img]

            logits = classifier.forward((x_smi, x_img))
            ps = torch.exp(logits)

            step_y_true.append(labels)
            step_y_score.append(ps)

        epoch_y_true = torch.cat(step_y_true)
        epoch_y_score = torch.cat(step_y_score)
        y_pred = torch.argmax(epoch_y_score, dim=1)

    return epoch_y_true.cpu().numpy(), epoch_y_score.cpu().numpy(), y_pred.cpu().numpy()
