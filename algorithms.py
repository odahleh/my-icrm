# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import division
from __future__ import print_function
import torch
import utils
import copy
import networks
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn as nn


ALGORITHMS = ['ERM', 'MultiInstancePredictor']

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, hparams):
        super(Algorithm, self).__init__()
        self.num_classes = num_classes
        self.hparams = hparams
        self.metrics = hparams['metrics']
        self.loss_func = utils.get_loss_function(hparams['loss'])
        self.device = hparams['device']
        self.n_tasks_per_step = hparams['n_sampled_tasks'] if hparams.get('n_sampled_tasks') else 0

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, num_classes, self.hparams)

        if hparams['is_parallel']:
            print('=> Using data parallel')
            self.featurizer = utils.data_parallel(self.featurizer)
            self.classifier = utils.data_parallel(self.classifier)

        self.network = torch.nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], self.network.parameters(), lr = self.hparams['lr'], weight_decay = self.hparams['weight_decay'])


    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x, y = None, model = None):
        raise NotImplementedError

    def evaluate(self, loader, weights = None, metrics = ['accuracy']):
        raise NotImplementedError()


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, hparams):
        super(ERM, self).__init__(input_shape, num_classes,
                                  hparams)

    def update(self, minibatches, unlabeled=None):
        self.network.train()
        if self.n_tasks_per_step != 0:
            indices = torch.randperm(len(minibatches))[:min(self.n_tasks_per_step, len(minibatches))]
            minibatches = [minibatches[i] for i in indices]

        all_y, p = [], []
        for _, (x,y) in enumerate(minibatches):
            out = self.predict(x, y)
            if isinstance(out, tuple):
                out, _ = out
            p.append(out); all_y.append(y)

        p = torch.cat(p, dim=0)
        all_y = torch.cat(all_y, dim=0).view(-1).long()
        loss = self.loss_func(p, all_y)

        # Compute performance metrics
        metric_results = utils.compute_metric(self.metrics, p, all_y)
        metric_results = {'train_' + key:val for key, val in metric_results.items()}
        metric_results['train_loss'] = loss.item()

        # Model parameters update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return metric_results

    def predict(self, x, y = None, model = None):
        return (self.network if model is None else model)(x)

    def _evaluate(self, model, loader, metrics=['accuracy', 'auroc']):
        metrics = metrics or self.metrics
        model.eval()
        result = {key: 0 for key in metrics}
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                p = self.predict(x, y, model=model)
                batch_results = utils.compute_metric(metrics, p, y)
                for metric in metrics:
                    result[metric] += batch_results[metric] * len(y)
                total += len(y)
        for metric in metrics:
            result[metric] /= (total + 1e-9) # type: ignore
        model.train()
        return result

    def evaluate(self, loader, n_test_samples = 100, module = 'train', cache = None):
        self.network.eval()
        result = {}
        metric_results = self._evaluate(self.network, loader, self.hparams['metrics'])
        result.update({f'{metric}': metric_results[metric] for metric in self.hparams['metrics']})
        self.network.train()
        return result

    def _get_ckpt_metric(self):
        return f'auroc'

class MultiInstancePredictor(ERM):
    """
        Multi Instance Predictor (MIP)
    """
    def __init__(self, input_shape, num_classes, hparams):
        self.context_len = hparams['context_length']
        hparams['is_transformer'] = 1
        super(MultiInstancePredictor, self).__init__(input_shape, num_classes,
                                  hparams)

    def predict(self, x, y, return_context = False, past_key_values = None):
        original_dims = x.ndim
        original_y_shape = y.shape

        # Handle input dimensions
        bs, ctxt, c, h, w = x.size()
        x = x.contiguous().view(bs * ctxt, c, h, w)
        x = self.featurizer(x)
        x = x.reshape(bs, ctxt, -1)

        outputs = self.classifier((x, y, None, past_key_values))
        p, past = outputs[0], outputs[1]

        return p.view(-1, p.size(-1)), past


    def _evaluate_robust(self, model, loader, metrics = ['accuracy', 'auroc'], test_cache = None):
        all_p, all_y = [], []
        result = {}
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            p, _ = self.predict(x, y)
            all_p.append(p.detach().cpu())
            all_y.append(y.detach().cpu())

        all_p = torch.cat(all_p, dim=0)
        all_y = torch.cat(all_y, dim=0)

        # Compute metrics with final tensors
        metric_results = utils.compute_metric(metrics, all_p, all_y)
        print(metric_results)
        result.update({f'{metric}': metric_results[metric] for metric in metrics})

        self.network.train()
        model.train()
        return result

    def evaluate(self, loader, module = 'train', cache = None):
        self.test_ctxt = list(range(0, 51, 5)) if module == 'test' else [0, 25, 50, 75, 100]
        result = self._evaluate_robust(self.network, loader, self.hparams['metrics'],  cache)
        return result
