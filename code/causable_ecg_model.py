# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:55:57 2021

@author: wes_c
"""

import logging
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import nn_define

logging.basicConfig(level=logging.INFO)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CausableECGModel:
    def __init__(self, generator_architecture='generator_lstm', 
                 classifier_architecture='discriminator_lstm',
                 generator_kwargs={}, classifier_kwargs={},
                 gen_state_dict_path='generator.pt',
                 classifier_state_dict_path='ecg_clf.pt',
                 random_forest_name='rf_clf.pkl'):
        #begin by loading the models
        self.generator = nn_define.NNFactory.get_nn(generator_architecture,
                                                    **generator_kwargs)
        self.generator.load_state_dict(torch.load(gen_state_dict_path))
        if not random_forest_name:
            self.clf = nn_define.NNFactory.get_nn(classifier_architecture,
                                                  **classifier_kwargs)
            self.clf.load_state_dict(torch.load(classifier_state_dict_path))
        else:
            self.clf = pickle.load(open(random_forest_name, 'rb'))
        self.rf = True
        self.counterfactuals = {}

    def _is_pathological(self, input):
        if not self.rf:
            return torch.squeeze(self.clf(input)).item() > 0.5
        else:
            return bool(self.clf.predict(input).item())

    def _get_ambles(self, input, start_index, replacing, amble_size=25):
        if start_index < amble_size:
            fill = amble_size - start_index
            fill = torch.zeros((1, 1, fill))
            preamble = torch.cat((fill, input[...,:start_index]), axis=2)
            postamble = input[...,(start_index+replacing):(start_index+replacing+amble_size)]
        elif start_index > input.shape[-1] - amble_size:
            fill = start_index - input.shape[-1] + amble_size
            fill = torch.zeros((1, 1, fill))
            preamble = input[...,(start_index-amble_size):start_index]
            postamble = torch.cat((input[...,start_index:], fill), axis=2)
        else:
            preamble = input[...,(start_index-amble_size):start_index]
            postamble = input[...,(start_index+replacing):(start_index+replacing+amble_size)]
        ambles = torch.cat((preamble, postamble), axis=2)
        return ambles

    def _fill_tensor(self, input, start_index, ambles):
        new_tensor = torch.clone(input)
        to_replace = self.generator(ambles)
        new_tensor[..., start_index:(start_index+to_replace.shape[-1])] = \
                                                                    to_replace
        return new_tensor

    def _counterfactual_generation(self, input, step_size=5,
                                   max_replace_size=250,
                                   min_replace_size=50,
                                   amble_size=25):
        assert max_replace_size/min_replace_size == max_replace_size//min_replace_size
        replacing = min_replace_size
        is_pathological = True
        while is_pathological and replacing <= max_replace_size:
            sequential_calls = replacing // min_replace_size
            for start in range(0, input.shape[-1] - sequential_calls*replacing,
                               step_size):
                new_tensor = torch.clone(input)
                for i in range(sequential_calls):
                    ambles = self._get_ambles(new_tensor, start + i*replacing,
                                              replacing=replacing,
                                              amble_size=amble_size)
                    new_tensor = self._fill_tensor(new_tensor,
                                                   start + i*replacing,
                                                   ambles)
                if self.rf:
                    is_pathological = self._is_pathological(new_tensor.detach().numpy().reshape(1, -1))
                else:
                    is_pathological = self._is_pathological(new_tensor)
                if not is_pathological:
                    break
            replacing += min_replace_size
        if not is_pathological:
            return (new_tensor, start, replacing)
        else:
            return None

    def _process_record(self, input, idx=''):
        if self._is_pathological(input):
            logging.info(f'Record {idx} is normal. Skipping')
        else:
            logging.info(f'Record {idx} is abnormal. Generating counterfactual.')
            if self.rf:
                input = torch.reshape(input, (1, 1, -1))
            counterfactual = self._counterfactual_generation(input)
            if not counterfactual:
                logging.warning(f'Could not generate counterfactual for record {idx}')
            else:
                logging.info(f'Successfully generated counterfactual for record {idx}')
            self.counterfactuals[idx] = counterfactual
        return

    def generate_counterfactuals(self, data):
        for idx in range(data.shape[0]):
            if not self.rf:
                input = torch.reshape(data[idx, ...], (1, 1, -1))
            else:
                input = data[idx,...].reshape(1, -1)
            self._process_record(input, idx=idx)
        return

    def plot_random(self, data, gen, pathologies, save=None):
        #get random counterfactual to plot
        plot_idx = np.random.choice(list(set(gen).intersection(set(pathologies))))
        new, start, replacing = self.counterfactuals[plot_idx]        
        fig, ax = plt.subplots()
        ax.plot(data[plot_idx,...])
        y_min, y_max = ax.get_ylim()
        ax.add_patch(Rectangle((start, y_min), replacing, y_max, facecolor='r', alpha=0.3))
        ax.set_title('Causal ECG Diagosis (Pathological)')
        if save:
            plt.savefig(save, dpi=500)
        return

    def analyze(self, data, labels, save='counterfactual_ecg.png'):
        assert self.counterfactuals
        pathologies = np.where(labels == 1)[0]
        gen = self.counterfactuals.keys()
        if not any(bool(val) for val in self.counterfactuals.values()):
            logging.warning('Analysis inaccurate, failed counterfactual generation.')
        gen_ct = len(set(gen).intersection(set(pathologies)))
        logging.info(f'Generated counterfactuals for {gen_ct} pathological ECGs.')
        logging.info(f'Total pathological ECGs: {len(set(pathologies))}.')
        self.plot_random(data, gen, pathologies, save=save)

if __name__ == '__main__':
    X_sample = pickle.load(open('X_sample.pkl', 'rb'))
    y_sample = pickle.load(open('y_sample.pkl', 'rb'))
    model = CausableECGModel()
    if device == 'cuda:0':
        X_sample = torch.cuda.FloatTensor(X_sample)
    else:
        X_sample = torch.FloatTensor(X_sample)
    model.generate_counterfactuals(X_sample)
    model.analyze(X_sample, y_sample)