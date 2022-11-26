import random

import cv2
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def randomize_smiles(m):
    try:
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)
    except:
        # https://github.com/rdkit/rdkit/issues/3086
        return Chem.MolToSmiles(m)


class SmilesDataset:
    def __init__(self, vocab, smi_df, randomize=False):
        self.vocab = vocab
        self.smi_df = smi_df

        self.randomize = randomize

        if self.randomize:
            self.mols = []
            for smi in self.smi_df['sm']:
                self.mols.append([Chem.MolFromSmiles(smiles) for smiles in smi.split(".")])

    def __len__(self):
        return len(self.smi_df)

    def __getitem__(self, index):
        if self.randomize:
            smiles_aug = ".".join([randomize_smiles(mol) for mol in self.mols[index]])
            return smiles_aug, self.smi_df['label'][index]
        else:
            return self.smi_df['sm'][index], self.smi_df['label'][index]

    def collate(self, batch):
        batch.sort(key=lambda tup: len(tup[0]), reverse=True)
        data, label = map(list, zip(*batch))

        tensors = [torch.tensor(self.vocab.string2ids(tokens, add_bos=True, add_eos=True),
                                dtype=torch.long)
                   for tokens in data]

        return tensors, torch.tensor(label)


class UnlabeledSmilesDataset:
    def __init__(self, vocab, smi_df):
        self.vocab = vocab
        self.smi_df = smi_df

    def __len__(self):
        return len(self.smi_df)

    def __getitem__(self, index):
        tokens = self.smi_df['smi'][index]
        # chemical compound id
        comp_id = self.smi_df['id'][index]

        return tokens, comp_id

    def collate(self, batch):
        # [(t, i), (t, i), (t, i), ]
        batch.sort(key=lambda tup: len(tup[0]), reverse=True)
        # input: sorted list of (,) tuples
        # output: separate lists, each with the one member of the tuple 
        data, comp_id = map(list, zip(*batch))

        tensors = [torch.tensor(self.vocab.string2ids(string, add_bos=True, add_eos=True), dtype=torch.long)
                   for string in data]

        return tensors, comp_id


class UnlabeledMultiRepDataset:
    def __init__(self, vocab, target_df, randomize=False, rotate=False):
        self.vocab = vocab
        self.target_df = target_df
        self.norm = transforms.Normalize(mean=mean, std=std)

        self.randomize = randomize
        self.rotate = rotate

        if self.randomize:
            self.mols = []
            for smi in self.target_df['smi']:
                self.mols.append([Chem.MolFromSmiles(smiles) for smiles in smi.split(".")])

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, index):
        if self.randomize:
            tokens = ".".join([randomize_smiles(mol) for mol in self.mols[index]])
        else:
            tokens = self.target_df['smi'][index]

        img_path = self.target_df['img_path'][index]

        img_arr = cv2.imread(img_path)
        angle = random.randint(0, 359)
        if angle > 0:
            rows, cols, channel = img_arr.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_arr = cv2.warpAffine(img_arr, rotation_matrix, (cols, rows), cv2.INTER_LINEAR,
                                     borderValue=(255, 255, 255))  # cv2.BORDER_CONSTANT, 255)

        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))

        img_tensor = torch.tensor(img_arr).type(torch.FloatTensor)
        img_tensor = self.norm(img_tensor)

        return tokens, img_tensor

    def collate(self, batch):
        batch.sort(key=lambda tup: len(tup[0]), reverse=True)
        data, img_tensor = map(list, zip(*batch))

        tensors = [torch.tensor(self.vocab.string2ids(string, add_bos=True, add_eos=True), dtype=torch.long)
                   for string in data]

        return tensors, torch.stack(img_tensor)


class MultiRepDataset:
    def __init__(self, vocab, target_df, randomize=False, rotate=False):
        self.vocab = vocab
        self.target_df = target_df
        self.norm = transforms.Normalize(mean=mean, std=std)

        self.randomize = randomize
        self.rotate = rotate

        if self.randomize:
            self.mols = []
            for smi in self.target_df['sm']:
                self.mols.append([Chem.MolFromSmiles(smiles) for smiles in smi.split(".")])

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, index):
        if self.randomize:
            tokens = ".".join([randomize_smiles(mol) for mol in self.mols[index]])
        else:
            tokens = self.target_df['sm'][index]

        img_path = self.target_df['img_path'][index]

        img_arr = cv2.imread(img_path)

        angle = random.randint(0, 359)
        if angle > 0:
            rows, cols, channel = img_arr.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_arr = cv2.warpAffine(img_arr, rotation_matrix, (cols, rows), cv2.INTER_LINEAR,
                                     borderValue=(255, 255, 255))  # cv2.BORDER_CONSTANT, 255)

        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))

        img_tensor = torch.tensor(img_arr).type(torch.FloatTensor)
        img_tensor = self.norm(img_tensor)

        return tokens, img_tensor, self.target_df['label'][index]

    def collate(self, batch):
        batch.sort(key=lambda tup: len(tup[0]), reverse=True)
        data, img_tensor, label = map(list, zip(*batch))

        tensors = [torch.tensor(self.vocab.string2ids(string, add_bos=True, add_eos=True), dtype=torch.long)
                   for string in data]

        return tensors, torch.stack(img_tensor), torch.tensor(label)


def draw_image_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 1.5
    # https://www.rdkit.org/docs/source/rdkit.Chem.Draw.html
    return Draw.MolToImage(mol, size=(200, 200))


class SmilesGenPilImageDataset:
    def __init__(self, vocab, target_df):
        self.vocab = vocab
        self.target_df = target_df
        self.norm = transforms.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, index):
        tokens = self.target_df['smi'][index]
        pil_image = draw_image_from_smiles(tokens)

        img_tensor = None
        if pil_image:
            img_arr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) / 255.0
            img_arr = img_arr.transpose((2, 0, 1))
            img_tensor = torch.tensor(img_arr).type(torch.FloatTensor)
            img_tensor = self.norm(img_tensor)

        comp_id = self.target_df['id'][index]

        return tokens, img_tensor, comp_id

    def collate(self, batch):
        batch = [tup for tup in batch if tup[1] is not None]
        batch.sort(key=lambda tup: len(tup[0]), reverse=True)
        data, img_tensor, comp_id = map(list, zip(*batch))

        tensors = [torch.tensor(self.vocab.string2ids(string, add_bos=True, add_eos=True), dtype=torch.long)
                   for string in data]

        return tensors, torch.stack(img_tensor), comp_id
