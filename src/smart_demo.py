'''
SMaRt Dashboard using Streamlit
'''
import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader

from datasets import UnlabeledSmilesDataset, SmilesGenPilImageDataset, draw_image_from_smiles

plt.rcParams.update({'axes.labelsize': 15, 'axes.titlesize': 20,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10})

st.header("SMaRt: Toxicity Classification through SMILES and Multimodal Representation")
st.caption("A portal for chemical safety practitioners to conduct statistical analysis in multiple domains.")
st.caption("Open to add new classifiers for cross-checking domain-specific toxicity predictions.")


#
# get predicted class for the specified dataloader and show progress indicator
#
def get_predictions(classifier, test_loader, model_info, total_len, multi_modal=False, eval_device='cpu'):
    step_y_score = []
    id_list = []

    if total_len > 1:
        latest_status = st.empty()
        prog_bar = st.progress(0)

    with torch.no_grad():
        classifier.eval()
        classifier.to(eval_device)

        for batch in test_loader:
            if multi_modal:
                id, logits = handle_multirep_batch(classifier, eval_device, batch)
            else:
                id, logits = handle_smiles_batch(classifier, eval_device, batch)
            ps = torch.exp(logits)

            step_y_score.append(ps)
            id_list.extend(id)

            if total_len > 1:
                latest_status.text('%s %d done' % (model_info, len(id_list)))
                prog_bar.progress(1.0 * len(id_list) / total_len)

        epoch_y_score = torch.cat(step_y_score)
        y_pred = torch.argmax(epoch_y_score, dim=1)

    return id_list, epoch_y_score.cpu().numpy(), y_pred.cpu().numpy()


#
# get logits (log of softmax) from SmilesClassifier for a mini-batch
#
def handle_smiles_batch(classifier, eval_device, batch):
    x, id = batch
    if not isinstance(x, list):
        x = x.to(eval_device)
    else:
        x = [x_i.to(eval_device) for x_i in x]
    logits = classifier(x)
    return id, logits


#
# get logits (log of softmax) from MultiRepM2DClassifier for a mini-batch
#
def handle_multirep_batch(classifier, eval_device, batch):
    x_smi, x_img, id = batch
    if not isinstance(x_smi, list):
        x_smi = x_smi.to(eval_device)
    else:
        x_smi = [x_i.to(eval_device) for x_i in x_smi]
    if not isinstance(x_img, list):
        x_img = x_img.to(eval_device)
    else:
        x_img = [x_i.to(eval_device) for x_i in x_img]
    logits = classifier.forward((x_smi, x_img))
    return id, logits


#
# saved classifiers in local file system
# organized first by SMaRt version, then by domain-specific task
#
model_dir = '/app/mmm/models'
vae_classifiers = {
    'ClinTox': '%s/clintox/version_13_e4.ckpt' % model_dir,
    'COSMOS': '%s/cosmos-ttc/version_8_e10.ckpt' % model_dir,
    'Tox21': '%s/tox21/version_93_e16.ckpt' % model_dir,
    'ToxiM': '%s/ToxiM/version_26_e13.ckpt' % model_dir,
    'WHOP': '%s/WHOP/version_10_e17.ckpt' % model_dir,
}
contrastive_vae_classifiers = {
    'ClinTox': '%s/clintox/version_8_e4.ckpt' % model_dir,
    'COSMOS': '%s/cosmos-ttc/version_6_e14.ckpt' % model_dir,
    'Tox21': '%s/tox21/version_123_e19.ckpt' % model_dir,
    'ToxiM': '%s/ToxiM/version_22_e12.ckpt' % model_dir,
    'WHOP': '%s/WHOP/version_5_e16.ckpt' % model_dir,
}
multimodal_vae_classifiers = {
    'Tox21': '%s/tox21/version_116_e18.ckpt' % model_dir,
    'ToxiM': '%s/ToxiM/version_30_e15.ckpt' % model_dir,
}
all_classifiers = {
    'SMILES (VAE)': vae_classifiers,
    'SMILES (Contrastive VAE)': contrastive_vae_classifiers,
    'Multimodal (Contrastive VAE)': multimodal_vae_classifiers
}

#
# display summary performance for classifiers
#
vae_performance = {
    'Dataset': ['ClinTox', 'COSMOS', 'Tox21', 'ToxiM', 'WHOP'],
    'Accuracy': [0.9986, 0.8288, 0.7653, 0.9316, 0.5882],
    'ROC': [0.9999, 0.8668, 0.7977, 0.9755, 0.8548],
    'MCC': [0.9908, 0.6808, 0.4425, 0.8437, 0.4672]
}
contrastive_vae_performance = {
    'Dataset': ['ClinTox', 'COSMOS', 'Tox21', 'ToxiM', 'WHOP'],
    'Accuracy': [0.9959, 0.8468, 0.75, 0.9402, 0.5588],
    'ROC': [0.9998, 0.8826, 0.8017, 0.9760, 0.8427],
    'MCC': [0.9733, 0.7114, 0.4575, 0.8582, 0.4280]
}
multimodal_vae_performance = {
    'Dataset': ['Tox21', 'ToxiM'],
    'Accuracy': [0.7474, 0.9255],
    'ROC': [0.7990, 0.9765],
    'MCC': [0.4728, 0.8275]
}

all_performance = {
    'SMILES (VAE)': vae_performance,
    'SMILES (Contrastive VAE)': contrastive_vae_performance,
    'Multimodal (Contrastive VAE)': multimodal_vae_performance
}

#
# human readable label values for specific domains
#
binary_toxic_labels = ['Not Toxic', 'Toxic']
# whop_clz_labels = ['Class Ia', 'Class Ib', 'Class II', 'Class III', 'Class U', 'Class Ligand']
whop_clz_labels = ['Extremely hazardous', 'Highly hazardous', 'Moderately hazardous', 'Slightly hazardous',
                   'Unlikely to present acute hazard', 'Class Ligand']
cosmos_ttc_labels = ['Low (Class I)', 'Intermediate (Class II)', 'High (Class III)']


def main(argv):
    #
    # render controls in side bar
    #
    classifier_vers = ['SMILES (VAE)', 'SMILES (Contrastive VAE)', 'Multimodal (Contrastive VAE)']
    sel_classifier_ver = st.sidebar.selectbox('Select a SMaRt based classifier', classifier_vers)

    st.sidebar.write('Select one or more domain-specific tasks')
    classifier_clintox = classifier_cosmos = classifier_tox21 = classifier_toxim = classifier_who = False
    if 'ClinTox' in all_classifiers[sel_classifier_ver]:
        classifier_clintox = st.sidebar.checkbox('ClinTox')
    if 'COSMOS' in all_classifiers[sel_classifier_ver]:
        classifier_cosmos = st.sidebar.checkbox('COSMOS')
    if 'Tox21' in all_classifiers[sel_classifier_ver]:
        classifier_tox21 = st.sidebar.checkbox('Tox21')
    if 'ToxiM' in all_classifiers[sel_classifier_ver]:
        classifier_toxim = st.sidebar.checkbox('ToxiM')
    if 'WHOP' in all_classifiers[sel_classifier_ver]:
        classifier_who = st.sidebar.checkbox('WHO Pesticides')

    input_smiles = st.sidebar.text_area('Enter SMILES')
    uploaded_file = st.sidebar.file_uploader("Or, upload a test data file", type=('tsv', 'csv', 'txt'))

    # show performance summary
    st.header(sel_classifier_ver)
    show_smmart_performance = st.checkbox('Show performance metrics')
    if show_smmart_performance:
        st.dataframe(pd.DataFrame(all_performance[sel_classifier_ver]))

    if st.sidebar.button('Submit'):
        multi_modal = (sel_classifier_ver == 'Multimodal (Contrastive VAE)')

        # gather multi-selected classifiers and load the checkpoints
        model_info_list = []
        if classifier_clintox:
            tox21_classifier = torch.load(all_classifiers[sel_classifier_ver]['ClinTox'])
            model_info_list.append(('ClinTox', tox21_classifier))
        if classifier_cosmos:
            tox21_classifier = torch.load(all_classifiers[sel_classifier_ver]['COSMOS'])
            model_info_list.append(('COSMOS', tox21_classifier))
        if classifier_tox21:
            tox21_classifier = torch.load(all_classifiers[sel_classifier_ver]['Tox21'])
            model_info_list.append(('Tox21', tox21_classifier))
        if classifier_toxim:
            toxim_classifier = torch.load(all_classifiers[sel_classifier_ver]['ToxiM'])
            model_info_list.append(('ToxiM', toxim_classifier))
        if classifier_who:
            toxim_classifier = torch.load(all_classifiers[sel_classifier_ver]['WHOP'])
            model_info_list.append(('WHO Pesticides', toxim_classifier))

        if len(model_info_list) == 0:
            st.warning("Please select one or more classifiers first")
            return

        single_mode = False

        # 1. take single input if present
        if input_smiles:
            if multi_modal:
                # error check for the single input SMILES
                try:
                    pil_image = draw_image_from_smiles(input_smiles)
                except:
                    st.warning('Error in drawing image')
                    return

            single_mode = True
            test_df = pd.DataFrame({
                'smi': [input_smiles],
                'id': ['single']
            })
        # 2. or take uploaded file
        elif uploaded_file:
            st.write('Processing uploaded file %s ...' % uploaded_file)
            test_df = pd.read_csv(uploaded_file)
            test_df = test_df.rename(columns={"sm": "smi"})
        else:
            st.warning('Please enter SMILES or upload a test data file')
            return

        #
        # get prediction for each selected classifier
        #
        for model_info in model_info_list:
            classifier = model_info[1]
            vocab = pickle.load(open(classifier.hparams.vocab, 'rb'))

            if multi_modal:
                test_dataset = SmilesGenPilImageDataset(vocab=vocab, target_df=test_df)
            else:
                test_dataset = UnlabeledSmilesDataset(vocab=vocab, smi_df=test_df)

            test_loader = DataLoader(
                test_dataset,
                num_workers=cpu_count(),
                batch_size=16,
                shuffle=False,
                collate_fn=test_dataset.collate
            )

            # this is the meat of the prediction
            id_list, all_ps, all_preds = get_predictions(classifier=classifier,
                                                         test_loader=test_loader,
                                                         total_len=len(test_dataset),
                                                         model_info=model_info[0],
                                                         multi_modal=multi_modal)

            #
            # result is scrambled, now restore the original order based on id mapping
            #
            id2index = {val: idx for idx, val in enumerate(id_list)}

            prediction, probability = [], []
            for id in test_df['id']:
                if id in id2index:
                    if model_info[0] == 'WHO Pesticides':
                        prediction.append(whop_clz_labels[all_preds[id2index[id]]])
                    elif model_info[0] == 'COSMOS':
                        prediction.append(cosmos_ttc_labels[all_preds[id2index[id]]])
                    else:
                        prediction.append(binary_toxic_labels[all_preds[id2index[id]]])
                    probability.append(all_ps[id2index[id], all_preds[id2index[id]]])
                else:
                    # it is possible for failure, and it will be missing, so handle it here
                    prediction.append('Unknown')
                    probability.append(0.0)
            test_df['%s prediction' % model_info[0]] = prediction
            test_df['%s probability' % model_info[0]] = probability

        if single_mode:
            col1, mid, col3 = st.columns([5, 1, 4])

            for model_info in model_info_list:
                for column_name in ['prediction', 'probability']:
                    column_name = '%s %s' % (model_info[0], column_name)
                    with col1:
                        col1.text(column_name)
                    with col3:
                        col3.text(test_df[column_name][0])

            # render the 2D drawing
            try:
                pil_image = draw_image_from_smiles(input_smiles)
                st.image(pil_image)
            except:
                st.warning('Error in drawing image')
        else:
            style_map = {
                '%s probability' % model_info[0]: '{:.4f}' for model_info in model_info_list
            }
            st.write("Scroll to the right for all the predictions")
            st.dataframe(test_df.style.format(style_map))


main(sys.argv)
