# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:56:00 2021

@author: wes_c
"""

import pickle
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import nn_define
import nn_train
import classifier_train

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

def main(gen_architecture='generator_lstm', disc_architecture='discriminator_lstm',
         gen_kwargs={}, disc_kwargs={}, preprocess_kwargs={}, train_kwargs={},
         device=device, gen_save=None, disc_save=None, classifier_kwargs={},
         rf=None, rf_kwargs={}):
    netG = nn_define.NNFactory.get_nn(gen_architecture, **gen_kwargs).to(device)
    netD = nn_define.NNFactory.get_nn(disc_architecture, **disc_kwargs).to(device)
    true_vals, ambles, fixed_t, fixed_a, X_train, X_test, y_train, y_test = nn_train.data_process(device=device, **preprocess_kwargs)
    if train_kwargs:
        netG, netD = nn_train.training_loop(netD, netG, true_vals, ambles, fixed_a,
                                            device=device, **train_kwargs)
    if isinstance(gen_save, str):
        torch.save(netG.state_dict(), gen_save)
    if isinstance(disc_save, str):
        torch.save(netD.state_dict(), disc_save)
    if rf:
        clf = RandomForestClassifier(**rf_kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))
        with open(rf, 'wb') as f:
            pickle.dump(clf, f)
    elif classifier_kwargs:
        classifier_train.train_classifier(X_train, y_train, X_test, y_test, **classifier_kwargs)
    return

if __name__ == '__main__':
    main(gen_kwargs={'input_size':50, 'hidden_states':16, 'out_size':50},
         disc_kwargs={'input_size':100, 'hidden_states':16},
         preprocess_kwargs={'generate_size':50, 'amble':25, 'step_size':10},
         train_kwargs={'num_epochs':1,
                  'batch_size':64, 'num_workers':0, 'ngpu':1,
                  'learning_rate':1e-5, 'beta1':0.45, 'beta2':0.997,
                  'verbose':True, 'fig_savename':'test_lstm_gan.png',
                  'animation_savename':'test_lstm_animation.gif'},
        classifier_kwargs={'device':device, 'plot_savename':'classifier_loss.png'},
        gen_save='generator.pt', disc_save='discriminator.pt',
        rf='rf_clf.pkl', rf_kwargs={})