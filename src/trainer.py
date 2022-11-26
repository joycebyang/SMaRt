import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.cloud_io import load as pl_load
from torch.optim import Adam

from metrics import calc_metrics
from models import VAEEncoder, Vanilla2DDrawingEncoder, M2DDrawingEncoder
from utils import maybe_dict_to_namespace


class MolPropClassifier(pl.LightningModule):
    def __init__(self, hparams):
        hparams = maybe_dict_to_namespace(hparams)
        super(MolPropClassifier, self).__init__()
        self.save_hyperparameters(hparams)

        self.model = None
        self.criterion = nn.CrossEntropyLoss()

        self.step_y_true = []
        self.step_y_score = []

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def shared_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        ps = torch.exp(logits)

        if self.criterion:
            loss = self.criterion(logits, y)
        else:
            loss = F.cross_entropy(ps, y)

        self.step_y_true.append(y)
        self.step_y_score.append(ps.detach())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('train_loss', loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        return loss

    def shared_epoch_end(self):
        epoch_y_true = torch.cat(self.step_y_true)
        epoch_y_score = torch.cat(self.step_y_score)

        epoch_metrics = calc_metrics(preds=epoch_y_score, target=epoch_y_true)

        self.step_y_true = []
        self.step_y_score = []

        return epoch_metrics

    def training_epoch_end(self, outputs):
        epoch_metrics = self.shared_epoch_end()

        self.log('avg_train_acc', epoch_metrics['accuracy'], prog_bar=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()

        epoch_metrics = self.shared_epoch_end()

        self.log('avg_val_acc', epoch_metrics['accuracy'], prog_bar=True)
        self.log('avg_val_loss', avg_loss, prog_bar=True)

        if 'auroc' in epoch_metrics:
            self.log('avg_val_auroc', epoch_metrics['auroc'])

        self.log('avg_val_f1', epoch_metrics['f1_score'])
        self.log('avg_val_mcc', epoch_metrics['matthews_corrcoef'])
        self.log('avg_val_precision', epoch_metrics['precision_recall'][0])
        self.log('avg_val_recall', epoch_metrics['precision_recall'][1])

        # this is to populate the Tensorboard metrics in the hparams plugin
        self.logger.log_metrics(metrics={'hp_metric': epoch_metrics['accuracy']}, step=self.current_epoch)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        epoch_metrics = self.shared_epoch_end()

        self.log('avg_test_acc', epoch_metrics['accuracy'])
        self.log('avg_test_loss', avg_loss)

        if 'auroc' in epoch_metrics:
            self.log('avg_test_auroc', epoch_metrics['auroc'])

        self.log('avg_test_f1', epoch_metrics['f1_score'])
        self.log('avg_test_mcc', epoch_metrics['matthews_corrcoef'])
        self.log('avg_test_precision', epoch_metrics['precision_recall'][0])
        self.log('avg_test_recall', epoch_metrics['precision_recall'][1])


class SmilesClassifier(MolPropClassifier):
    def __init__(self, hparams):
        hparams = maybe_dict_to_namespace(hparams)
        super(SmilesClassifier, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        vocab = pickle.load(open(hparams.vocab, 'rb'))

        self.model = VAEEncoder(vocab=vocab,
                                hidden_size=hparams.q_d_h,
                                num_layers=hparams.q_n_layers,
                                bidirectional=hparams.q_bidir,
                                dropout=hparams.q_dropout)

        q_d_last = hparams.q_d_h * (2 if hparams.q_bidir else 1)
        self.fc_out = nn.Linear(q_d_last,
                                hparams.num_classes)

    def forward(self, x):
        features = self.model.forward(x)
        logits = self.fc_out(features)

        return F.log_softmax(logits, dim=1)


class MultiRepClassifier(MolPropClassifier):
    def __init__(self, hparams):
        hparams = maybe_dict_to_namespace(hparams)
        super(MultiRepClassifier, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        vocab = pickle.load(open(hparams.vocab, 'rb'))

        self.smi_model = VAEEncoder(vocab=vocab,
                                    hidden_size=hparams.q_d_h,
                                    num_layers=hparams.q_n_layers,
                                    bidirectional=hparams.q_bidir,
                                    dropout=hparams.q_dropout)

        self.img_model = None

        q_d_last = hparams.q_d_h * (2 if hparams.q_bidir else 1)
        fused_enc_dim = q_d_last + hparams.enc_out_dim

        self.predictor = nn.Sequential(
            nn.Linear(fused_enc_dim, fused_enc_dim),
            nn.LayerNorm(fused_enc_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=hparams.final_drop_rate),
        )
        self.fc_out = nn.Linear(fused_enc_dim,
                                hparams.num_classes)

    def shared_step(self, batch):
        x_smi, x_img, y = batch

        logits = self.forward((x_smi, x_img))
        ps = torch.exp(logits)

        if self.criterion:
            loss = self.criterion(logits, y)
        else:
            loss = F.cross_entropy(ps, y)

        self.step_y_true.append(y)
        self.step_y_score.append(ps.detach())

        return loss

    def forward(self, x):
        x_smi, x_img = x
        smi_features = self.smi_model.forward(x_smi)
        img_features = self.img_model.forward(x_img)
        features = torch.cat((smi_features, img_features), 1)
        features = self.predictor(features)

        logits = self.fc_out(features)

        return F.log_softmax(logits, dim=1)


class MultiRepVanilla2DClassifier(MultiRepClassifier):
    def __init__(self, hparams):
        hparams = maybe_dict_to_namespace(hparams)
        super(MultiRepVanilla2DClassifier, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        self.img_model = Vanilla2DDrawingEncoder(enc_out_dim=hparams.enc_out_dim)


class MultiRepM2DClassifier(MultiRepClassifier):
    def __init__(self, hparams):
        hparams = maybe_dict_to_namespace(hparams)
        super(MultiRepM2DClassifier, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        self.img_model = M2DDrawingEncoder(hidden_size=hparams.enc_out_dim,
                                           drop_rate=hparams.image_drop_rate)


def load_custom_classifier_from_checkpoint(checkpoint_path, class_to_instantiate: MolPropClassifier,
                                           strict=False, map_location=None) -> MolPropClassifier:
    """
    @param checkpoint_path: url or file path to the classifier checkpoint
    @param strict: passed to load_state_dict
    @param map_location: 'cuda' if GPU is available, otherwise 'cpu'
    @param class_to_instantiate: subclass of MolPropClassifier
    @return:
    """
    checkpoint = pl_load(checkpoint_path, map_location=map_location)
    classifier = class_to_instantiate(checkpoint['hyper_parameters'])
    classifier.load_state_dict(checkpoint['state_dict'], strict=strict)
    return classifier
