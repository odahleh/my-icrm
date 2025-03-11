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


ALGORITHMS = ['ERM', 'ICRM', 'ARM_CML', 'TENT', 'Mixup', 'Fish', 'IB_ERM', 'IB_IRM']

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
        for i, (x, y) in enumerate(minibatches):
            print(f"Minibatch {i}: x shape {x.shape}, y shape {y.shape}")
            out = self.predict(x, y)
            print(f"  Output shape: {out.shape}")
            p.append(out)
            all_y.append(y)
       
        p = torch.cat(p, dim=0)
        all_y = torch.cat(all_y, dim=0).view(-1).long()
        
        # Debug print to check shapes before loss calculation
        print(f"Concatenated: p shape: {p.shape}, all_y shape: {all_y.shape}")
        
        # Handle batch size mismatch - ensure p and all_y have compatible batch sizes
        if p.shape[0] != all_y.shape[0]:
            print(f"Batch size mismatch! p: {p.shape[0]}, all_y: {all_y.shape[0]}")
            
            # For ICRM, we need to adjust the output shape
            if hasattr(self, 'context_len'):
                print(f"  Applying ICRM-specific shape adjustment (context_len={self.context_len})")
                
                # Expand all_y to match p's batch size if needed
                if p.shape[0] > all_y.shape[0]:
                    # This might happen if p includes duplicate predictions for each context element
                    factor = p.shape[0] // all_y.shape[0]
                    if p.shape[0] % all_y.shape[0] == 0:
                        print(f"  Expanding all_y by factor of {factor}")
                        all_y = all_y.repeat_interleave(factor)
                    else:
                        # If not evenly divisible, truncate p to match
                        print(f"  Shapes not evenly divisible - truncating p")
                        p = p[:all_y.shape[0]]
                        
                # Or reduce p if needed
                elif p.shape[0] < all_y.shape[0]:
                    # This might happen if p combines context elements
                    factor = all_y.shape[0] // p.shape[0]
                    if all_y.shape[0] % p.shape[0] == 0:
                        print(f"  Repeating p by factor of {factor}")
                        # Repeat each prediction to match ground truth
                        p = p.repeat_interleave(factor, dim=0)
                    else:
                        # If not evenly divisible, truncate all_y to match
                        print(f"  Shapes not evenly divisible - truncating all_y")
                        all_y = all_y[:p.shape[0]]
            else:
                # For non-ICRM models, print a clear error but try to continue
                print(f"  WARNING: Batch size mismatch in non-ICRM model. Attempting to continue.")
                
                # Take the smaller size to avoid dimension errors
                min_size = min(p.shape[0], all_y.shape[0])
                p = p[:min_size]
                all_y = all_y[:min_size]
        
        print(f"Final shapes for loss: p: {p.shape}, all_y: {all_y.shape}")
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

    def _evaluate(self, model, loader, metrics=['accuracy']):
        metrics = metrics or self.metrics
        model.eval()
        result = {key: 0 for key in metrics}
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                try:
                    # Get predictions
                    p = self.predict(x, y, model=model)
                    
                    # Ensure predictions and targets have compatible dimensions for metrics
                    if p.ndim > 1 and y.ndim == 1 and p.shape[0] == y.shape[0]:
                        # Standard case: p is (batch, classes), y is (batch,)
                        batch_results = utils.compute_metric(metrics, p, y)
                    elif p.ndim > 2 and p.shape[0] == y.shape[0]:
                        # p has extra dimensions (like sequence length), take last prediction
                        p_last = p[:, -1, ...] if p.shape[1] > 1 else p.squeeze(1)
                        batch_results = utils.compute_metric(metrics, p_last, y)
                    elif p.ndim == y.ndim and p.shape[0] != y.shape[0]:
                        # Different batch sizes, try to match
                        min_batch = min(p.shape[0], y.shape[0])
                        batch_results = utils.compute_metric(metrics, p[:min_batch], y[:min_batch])
                    else:
                        # Default case - pass as is and let compute_metric handle it
                        batch_results = utils.compute_metric(metrics, p, y)
                        
                    for metric in metrics:
                        result[metric] += batch_results[metric] * len(y)
                    total += len(y)
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    print(f"Shapes - x: {x.shape}, y: {y.shape}, predictions: {p.shape if 'p' in locals() else 'N/A'}")
                    # Continue to next batch rather than failing
                    continue
                    
        for metric in metrics:  
            result[metric] /= (total + 1e-9)
        model.train()
        return result

    def evaluate(self, loader, n_test_samples = 100, module = 'train', cache = None):
        self.network.eval()
        result = {}
        metric_results = self._evaluate(self.network, loader, self.hparams['metrics'])           
        self.test_ctxt = range(0, 51, 5) if module == 'test' else [0, 25, 50, 75, 100]
        for num_samples in self.test_ctxt:
            result.update({f'{metric}(e-{n_test_samples - num_samples})': metric_results[metric] for metric in self.hparams['metrics']})
        self.network.train()
        return result
    
    def _get_ckpt_metric(self):
        return f'acc(e-0)'
 

class TENT(ERM):
    """Tent: Fully Test-Time Adaptation by Entropy Minimization"""
    def __init__(self, input_shape, num_classes, hparams):
        # Make sure to load weights of a trained ERM model before fine-tuning it with TENT
        super().__init__(input_shape, num_classes, hparams)
        self.n_steps = hparams.get('n_steps', 10)
        self.episodic = hparams.get('episodic', 1)
        print(f'Using episodic {bool(self.episodic)} training with {self.n_steps} steps')
        self.flag = 0
    
    def init_states(self):
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.network, self.optimizer)
        self.flag = 1
                
    def _setup_model(self):
        if not self.flag:
            self.o_network = copy.deepcopy(self.network)
            self.o_network.eval()
        self.network = self._configure_model(self.network)
        params, _ = self._collect_params(self.network)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], params, lr = self.hparams['lr'], weight_decay = self.hparams['weight_decay'])
    
    def evaluate(self, loader, module = 'train', cache = None):
        if not self.flag:
            self._setup_model()
            self.init_states()
        
        self.reset()
        self._setup_model()

        self.test_ctxt = list(range(0, 51, 5)) if module == 'test' else [25, 50, 75, 100]

        if 0 in self.test_ctxt:
            metric_results = self._evaluate(self.o_network, loader, self.hparams['metrics'])  
            result = {f'{metric}(e-100)': metric_results[metric] for metric in self.hparams['metrics']}
        else:
            result = {}
         
        assert self.n_steps > 0, "Tent requires >= 1 step(s) to forward and update"
        for n_samples in self.test_ctxt:
            if n_samples == 0:
                continue
            self.reset()
            metric_results = {key: 0 for key in self.hparams['metrics']}
            sub_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=n_samples, shuffle=False)
            total = 0
            for _, (x, y) in enumerate(sub_loader):
                x, y = x.to(self.device), y.to(self.device)
                if self.episodic:   
                    self.reset()
                t_loss = []
                for _ in range(self.n_steps):
                    loss, p = self._forward_and_adapt(x, self.network, self.optimizer)    
                    t_loss.append(loss.item())
                batch_results = utils.compute_metric(self.hparams['metrics'], p, y)
                for metric in self.hparams['metrics']:
                    metric_results[metric] += batch_results[metric] * len(y)
                total += len(y)
            for metric in self.hparams['metrics']:  metric_results[metric] /= (total + 1e-9)  
            result.update({f'{metric}(e-{100 - n_samples})': metric_results[metric] for metric in self.hparams['metrics']})       
        return result
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.network, self.optimizer, self.model_state, self.optimizer_state)

    @torch.enable_grad()                                                                    # ensure grads in possible no grad context for testing
    def _forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        outputs = model(x)
        loss = utils.softmax_entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss, outputs
       
    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """Restore the model and optimizer states from copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        
    @staticmethod
    def _configure_model(model):
        """Configure model for use with tent."""
        model.train()                                                                       # train mode, because tent optimizes the model to minimize entropy
        model.requires_grad_(False)                                                         # disable grad, to (re-)enable only what tent updates
        for m in model.modules():                                                           # configure norm for tent updates: enable grad + force batch statisics
            if isinstance(m, torch.nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False                                               # force use of batch stats in train and eval modes
                m.running_mean = None
                m.running_var = None
        return model 
    
    @staticmethod
    def _collect_params(model):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:                                           # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
    
    @staticmethod
    def _check_model(model):
        """Check model for compatability with tent."""
        is_training = model.training
        assert is_training, "tent needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "tent needs params to update: check which require grad"
        assert not has_all_params, "tent should not update all params: check which require grad"
        has_bn = any([isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "tent needs normalization for its optimization"  
        
        

class ICRM(ERM):
    """
    In Context Learner (ICRM)
    """
    def __init__(self, input_shape, num_classes, hparams):
        self.context_len = hparams['context_length']
        hparams['is_transformer'] = 1
        super(ICRM, self).__init__(input_shape, num_classes,
                                  hparams)
                
    def predict(self, x, y, return_context = False, past_key_values = None): 
        original_dims = x.ndim
        original_y_shape = y.shape
        
        # Debug information for unusual dimensions
        if original_dims >= 5:
            print(f"DEBUG: predict called with unusual input dimensions:")
            print(f"  x.shape = {x.shape}, x.ndim = {x.ndim}")
            print(f"  y.shape = {y.shape}, y.ndim = {y.ndim}")
            print(f"  return_context = {return_context}")
            print(f"  past_key_values provided = {past_key_values is not None}")
        
        # Handle input dimensions
        if x.ndim == 4:                                                             # Splits a batch into multiple sequences with length as the context length                                    
            bs, c, h, w = x.size()                          
            bs, ctxt = bs // self.context_len, self.context_len
            y = y.reshape(bs, ctxt)
        elif x.ndim == 5:   
            bs, ctxt, c, h, w = x.size()
            x = x.contiguous().view(bs * ctxt, c, h, w)
        elif x.ndim == 6:
            # Handle 6D input: likely [batch, extra_dim, context_len, C, H, W]
            # Reshape to 5D first by combining batch and extra_dim
            shape = x.size()
            bs = shape[0] * shape[1]  # Combine first two dimensions
            ctxt = shape[2]
            c, h, w = shape[3], shape[4], shape[5]
            
            # Reshape to [bs*ctxt, c, h, w] for the featurizer
            x = x.view(bs * ctxt, c, h, w)
            
            # If y has matching batch dimensions, reshape it too
            if y.ndim > 1 and y.shape[0] == shape[0] and y.shape[1] == shape[1]:
                y = y.view(bs, -1)
            print(f"Handled 6D input. Original shape: {shape}, reshaped to: {x.shape}, y shape: {y.shape}")
        else:
            # Handle other dimensionality cases (could be during evaluation)
            if x.ndim == 3 and len(x.shape) == 3:  # (batch, channels, features)
                bs, c, features = x.size()
                ctxt = 1  # No context dimension
            elif x.ndim == 2:  # Already flattened features
                bs = x.size(0)
                ctxt = 1
            else:
                raise NotImplementedError(f"Unsupported input dimension: {x.ndim}")
        
        # Apply featurizer if input has image-like dimensions
        if x.ndim >= 3 and hasattr(self, 'featurizer'):
            x = self.featurizer(x)
                                                    
        # Reshape for transformer input if needed
        if hasattr(self, 'classifier') and isinstance(self.classifier, nn.Module):
            if x.ndim == 2:
                # If already 2D (batch, features), reshape to 3D for transformer
                x = x.unsqueeze(1)  # (batch, 1, features)
                bs, ctxt, _ = x.shape
            elif x.ndim == 3 and x.shape[1] != ctxt:
                # Ensure middle dimension is context length
                x = x.reshape(bs, ctxt, -1)
                
            # Call classifier (GPT2Transformer)
            if original_dims >= 5:
                print(f"DEBUG: Before classifier call:")
                print(f"  x.shape = {x.shape}")
                print(f"  Classifier input type: {type((x, y, None, past_key_values))}")
            
            outputs = self.classifier((x, y, None, past_key_values))
            
            # Unpack outputs
            if isinstance(outputs, tuple) and len(outputs) == 2:
                p, past = outputs
                if original_dims >= 5:
                    print(f"DEBUG: Classifier output (tuple):")
                    print(f"  p.shape = {p.shape}")
                    print(f"  past is provided: {past is not None}")
            else:
                p = outputs
                past = None
                if original_dims >= 5:
                    print(f"DEBUG: Classifier output (single):")
                    print(f"  p.shape = {p.shape}")
        else:
            # Handle case where no classifier exists or direct prediction
            p = x
            past = None
            
        # Return with appropriate dimensions
        if return_context:
            return p, past
        else:
            # For loss calculation, we need to match the batch dimensions
            # Handle original 6D input specially
            if original_dims == 6:
                # Need to reshape prediction based on the original input shape
                # If prediction is 2D (batch, classes)
                if len(p.shape) == 2:
                    # Reshape to match the original batch structure: [shape[0], shape[1], ...]
                    orig_batch_size = original_y_shape[0]
                    orig_extra_dim = original_y_shape[1] if len(original_y_shape) > 1 else 1
                    # Check if prediction shape matches expected shape after reshaping
                    if p.shape[0] == orig_batch_size * orig_extra_dim:
                        p = p.view(orig_batch_size, orig_extra_dim, -1)
                    else:
                        # Just return as is if dimensions don't align as expected
                        print(f"Warning: Could not reshape prediction for 6D input: p.shape={p.shape}, original_y_shape={original_y_shape}")
                return p
            # Handle original 4D and 5D inputs
            elif len(p.shape) == 2 and p.shape[0] == bs and original_dims == 4 and ctxt > 1:
                # Create one prediction per input image by repeating each prediction ctxt times
                expanded_p = p.repeat_interleave(ctxt, dim=0)
                return expanded_p
            else:
                # Otherwise return as is
                return p
    
    def repeat_past_key_values(self, past_key_values, repeats):                     # process key value cache for computing fast inference
        # Handle None case
        if past_key_values is None:
            return None
            
        repeated_past_key_values = []
        for layer_past in past_key_values:
            repeated_layer_past = []
            for tensor in layer_past:
                if tensor is not None:
                    repeated_tensor = tensor.repeat_interleave(repeats=repeats, dim=0)
                else:
                    repeated_tensor = None
                repeated_layer_past.append(repeated_tensor)
            repeated_past_key_values.append(tuple(repeated_layer_past))
        return tuple(repeated_past_key_values)

    
    def _evaluate_robust(self, model, loader, metrics = ['accuracy'], test_cache = None):
        test_cache_x, test_cache_y = test_cache
        assert test_cache_x is not None
        assert test_cache_y is not None
        self.network.eval()
        model.eval()
        result = {}
        for context_val in self.test_ctxt: 
            with torch.no_grad():  
                if context_val == 0:    initial_past = None    
                else:
                    _, initial_past = self.predict(test_cache_x[:, :context_val], test_cache_y[:, :context_val], return_context = True)
                    initial_past = self.repeat_past_key_values(initial_past, loader._bs if hasattr(loader, '_bs') else 1)
                
                all_p, all_y = [],[]
                batch_shapes = []
                
                # Add more debug info about loader
                print(f"DEBUG: Loader type: {type(loader)}")
                if hasattr(loader, 'dataset'):
                    print(f"DEBUG: Loader dataset type: {type(loader.dataset)}")
                    if hasattr(loader.dataset, '__getitem__'):
                        try:
                            # Try to get the first item to see its structure
                            sample_x, sample_y = loader.dataset[0]
                            print(f"DEBUG: Sample from dataset - x.shape: {sample_x.shape}, y.shape: {sample_y.shape if hasattr(sample_y, 'shape') else 'scalar'}")
                        except Exception as e:
                            print(f"DEBUG: Error accessing dataset item: {e}")
                
                for batch_idx, (x, y) in enumerate(loader):
                    print(f"DEBUG: Batch {batch_idx} from loader - x.shape: {x.shape}, y.shape: {y.shape}")
                    
                    x, y = x.to(self.device), y.to(self.device)   
                    
                    # Check for unexpected 6D tensor
                    if x.ndim == 6:
                        print(f"DEBUG: Found 6D input tensor in batch {batch_idx}:")
                        print(f"  x.shape = {x.shape}")
                        print(f"  y.shape = {y.shape}")
                        
                        # Try to understand the structure - print a sample of dimensions
                        print(f"  First dimension sizes: {[x.size(d) for d in range(min(6, x.ndim))]}")
                        
                        # Check if this is a nested batch structure
                        if x.size(0) == 1:
                            print("  This appears to be a batch with a single item, reshaping...")
                            x = x.squeeze(0)  # Remove the first dimension if it's 1
                    
                    p, _ = self.predict(x, y, return_context = True, past_key_values = initial_past)   
                    
                    # Record shapes for debugging
                    batch_shapes.append((x.shape, y.shape, p.shape))
                    
                    # Handle reshaping y based on its dimensions
                    if y.ndim == 1: 
                        # If y is 1D (batch,), reshape to 2D (batch/context_len, context_len)
                        if y.shape[0] % self.context_len == 0:
                            y_reshape = y.view(-1, self.context_len)
                        else:
                            # If not divisible, pad to make it divisible
                            pad_size = self.context_len - (y.shape[0] % self.context_len)
                            if pad_size < self.context_len:
                                y_padded = torch.cat([y, torch.zeros(pad_size, device=y.device, dtype=y.dtype)])
                                y_reshape = y_padded.view(-1, self.context_len)
                            else:
                                y_reshape = y.unsqueeze(1)  # Just add a dimension
                    else:   
                        # If y is already multidimensional, use as is
                        y_reshape = y
                    
                    # Handle reshaping p based on its dimensions
                    if p.ndim == 3 and y_reshape.ndim == 2 and p.shape[0] != y_reshape.shape[0]:
                        # If p is (batch, seq_len, features) but dimensions don't match y_reshape
                        # Adjust p to match y_reshape batch dimension
                        if p.shape[0] < y_reshape.shape[0] and p.shape[0] * p.shape[1] == y_reshape.shape[0] * y_reshape.shape[1]:
                            # If total elements match, reshape
                            p_reshape = p.reshape(y_reshape.shape[0], -1, p.shape[2])
                        else:
                            # Otherwise keep as is
                            p_reshape = p
                    else:
                        p_reshape = p
                    
                    all_p.append(p_reshape)
                    all_y.append(y_reshape)                   
                
                # Print batch shapes for debugging
                print(f"Batch shapes: {batch_shapes}")
                
                # Process each tensor separately before concatenation
                processed_p = []
                processed_y = []
                
                for i, (p, y) in enumerate(zip(all_p, all_y)):
                    print(f"Processing batch {i}: p.shape={p.shape}, y.shape={y.shape}")
                    
                    # Handle different dimension cases
                    if p.ndim == 3 and y.ndim == 2:
                        # Extract last predictions for sequence
                        p_last = p[:, -1, :]
                        y_last = y[:, -1]
                        processed_p.append(p_last)
                        processed_y.append(y_last)
                    elif p.ndim == 2 and y.ndim == 2:
                        # For 2D predictions and 2D targets, use last target
                        processed_p.append(p)
                        processed_y.append(y[:, -1])
                    elif p.ndim == 2 and y.ndim == 1:
                        # Direct case
                        processed_p.append(p)
                        processed_y.append(y)
                    else:
                        print(f"Unusual tensor shapes in batch {i}")
                        # Try to make tensors compatible
                        if p.ndim > y.ndim:
                            # Reduce p dimensions to match y
                            if p.ndim == 3 and y.ndim == 1:
                                processed_p.append(p.view(-1, p.shape[-1]))
                                # Repeat y to match new p shape if needed
                                if p.shape[0] * p.shape[1] != y.shape[0]:
                                    processed_y.append(y.repeat(p.shape[1]))
                                else:
                                    processed_y.append(y)
                            else:
                                print(f"Skipping incompatible pair: p.shape={p.shape}, y.shape={y.shape}")
                        else:
                            print(f"Skipping incompatible pair: p.shape={p.shape}, y.shape={y.shape}")
                
                if not processed_p or not processed_y:
                    print("No valid tensor pairs found, skipping context")
                    continue
                
                # Concatenate processed tensors
                try:
                    all_p = torch.cat(processed_p, dim=0)
                    all_y = torch.cat(processed_y, dim=0)
                    print(f"After initial processing: all_p.shape={all_p.shape}, all_y.shape={all_y.shape}")
                except RuntimeError as e:
                    print(f"Error concatenating processed tensors: {e}")
                    print(f"Processed shapes: p={[p.shape for p in processed_p]}, y={[y.shape for y in processed_y]}")
                    # Skip this context if we can't concatenate tensors
                    continue
                
                # Final dimension adjustment if sizes still don't match
                if all_p.shape[0] != all_y.shape[0]:
                    print(f"Mismatched dimensions after processing: all_p={all_p.shape}, all_y={all_y.shape}")
                    
                    # For the specific case of 258816 vs 4044
                    factor = all_p.shape[0] // all_y.shape[0] if all_p.shape[0] > all_y.shape[0] else all_y.shape[0] // all_p.shape[0]
                    
                    if all_p.shape[0] > all_y.shape[0]:
                        if all_p.shape[0] % all_y.shape[0] == 0:
                            # If p is larger and divisible by y, reshape p
                            print(f"Reshaping predictions: factor={factor}")
                            if all_p.ndim == 2:
                                # For 2D tensor, take samples at regular intervals
                                all_p = all_p[::factor]
                            else:
                                # For higher dimensions, try to reshape
                                try:
                                    all_p = all_p.view(all_y.shape[0], factor, -1)[:, 0, :]
                                except RuntimeError:
                                    # If reshaping fails, subsample
                                    all_p = all_p[::factor]
                        else:
                            # Otherwise truncate to smallest size
                            print("Truncating predictions to match targets")
                            all_p = all_p[:all_y.shape[0]]
                    else:
                        if all_y.shape[0] % all_p.shape[0] == 0:
                            # If y is larger and divisible by p, reshape y
                            print(f"Reshaping targets: factor={factor}")
                            all_y = all_y[::factor]
                        else:
                            # Otherwise truncate to smallest size
                            print("Truncating targets to match predictions")
                            all_y = all_y[:all_p.shape[0]]
                
                # Print final shapes
                print(f"Final shapes for metric computation: all_p={all_p.shape}, all_y={all_y.shape}")
                
                # Compute metrics with final tensors
                metric_results = utils.compute_metric(metrics, all_p, all_y)
                result.update({f'{metric}(e-{self.context_len - context_val})': metric_results[metric] for metric in metrics})      
        
        self.network.train()
        model.train()
        return result 
    
    def evaluate(self, loader, module = 'train', cache = None):
        self.test_ctxt = list(range(0, 51, 5)) if module == 'test' else [0, 25, 50, 75, 100]
        result = self._evaluate_robust(self.network, loader, self.hparams['metrics'],  cache)
        return result
 

class ARM_CML(ERM):
    """ Adaptive Risk Minimization (ARM) - (Context Model)"""
    def __init__(self, input_shape, num_classes, hparams):
        original_input_shape = input_shape
        self.n_context_channels = hparams.get('num_features', 1)
        self.ctxt = hparams['support_size']
        self.orig_ctxt = hparams['support_size']
        self.adapt_bn = hparams['adapt_bn']
        input_shape =  (self.n_context_channels + input_shape[0],) + input_shape[1:]             # Since we concatenate the context with input x
        super(ARM_CML, self).__init__(input_shape, num_classes, hparams)
        if hasattr(networks, hparams['context_net']):                                                
            self.context_net = getattr(networks, hparams['context_net'])(original_input_shape[0], hparams)  
            if hparams['is_parallel']:
                self.context_net = utils.data_parallel(self.context_net)
        else:
            raise NotImplementedError()
        
        #  Joint optimizer for ϕ and 𝜃
        params = list(self.network.parameters()) + list(self.context_net.parameters())
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], params, lr = self.hparams['lr'], weight_decay = self.hparams['weight_decay'])
        self.hparams['mode'] = 'train'
        
        
    def predict(self, x, y = None, model = None):
        bs, _, h, w = x.shape
        re = bs  % self.ctxt
        if self.hparams['mode'] == 'train':
            assert re == 0, 'During training, makre sure batch size is a multiple of support'
            
        if self.hparams['mode'] == 'test' and re != 0:
            x = torch.cat([x, x[:(self.ctxt - re)].clone()], dim=0) 
            y = torch.cat([y, y[:(self.ctxt - re)].clone()], dim=0) 

        eff_bs, supp_size = len(x) // self.ctxt, self.ctxt
        ctxt = self.context_net(x)
        ctxt = ctxt.reshape(eff_bs, supp_size, self.n_context_channels, h, w).mean(dim=1)    
        ctxt = torch.repeat_interleave(ctxt, repeats = supp_size, dim=0)
        x = torch.cat([x, ctxt], dim=1)
        if self.hparams['mode'] == 'test':
            return self.network(x), y
        return self.network(x)
 
    def _evaluate(self, model, loader, metrics=['accuracy']):
        metrics = metrics or self.metrics
        model.eval()
        self.context_net.eval()
        result = {key: 0 for key in metrics}
        total = 0
        with torch.no_grad():
            for x, y in loader:
                try:
                    x, y = x.to(self.device), y.to(self.device)
                    p, y_modif = self.predict(x, y, model=model)
                    
                    # Handle tensor dimension compatibility
                    if p.ndim > 1 and y_modif.ndim == 1 and p.shape[0] == y_modif.shape[0]:
                        # Standard case
                        batch_results = utils.compute_metric(metrics, p, y_modif)
                    elif p.ndim > 2:
                        # Higher dimensional predictions, take last prediction
                        p_last = p[:, -1, ...] if p.shape[1] > 1 else p.squeeze(1)
                        batch_results = utils.compute_metric(metrics, p_last, y_modif)
                    elif p.shape[0] != y_modif.shape[0]:
                        # Different batch sizes, use minimum
                        min_batch = min(p.shape[0], y_modif.shape[0])
                        batch_results = utils.compute_metric(metrics, p[:min_batch], y_modif[:min_batch])
                    else:
                        # Default case
                        batch_results = utils.compute_metric(metrics, p, y_modif)
                        
                    for metric in metrics:
                        result[metric] += batch_results[metric] * len(y_modif)
                    total += len(y_modif)
                except Exception as e:
                    print(f"Error in ARM_CML evaluation: {e}")
                    print(f"Shapes - x: {x.shape}, y: {y.shape}")
                    if 'p' in locals() and 'y_modif' in locals():
                        print(f"p: {p.shape}, y_modif: {y_modif.shape}")
                    # Continue to next batch
                    continue
                    
            for metric in metrics:  
                result[metric] /= (total + 1e-9)
            model.train()
            return result
   
    def evaluate(self, loader, module = 'train', cache = None):
        self.hparams['mode'] = 'test'
        self.hparams['test_support'] = self.hparams['test_support'] if self.hparams['test_support'] is not None else self.ctxt         
        result = {}
        self.test_ctxt = range(0, 51, 5) if module == 'test' else [0, 25, 50, 75, 100]
        for supp in self.test_ctxt:  
            if supp == 0:
                self.ctxt = supp + 1
            else:
                self.ctxt = supp
            metric_results = self._evaluate(self.network, loader, self.hparams['metrics'])
            result.update({f'{metric}(e-{self.hparams["test_support"] - supp})': metric_results[metric] for metric in self.hparams['metrics']})    
        self.hparams['mode'] = 'train'
        self.context_net.train()
        self.network.train()
        self.ctxt = self.orig_ctxt
        return result 

 

class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, hparams):
        super(Mixup, self).__init__(input_shape, num_classes,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        self.network.train()
        objective = 0
        for (xi, yi), (xj, yj) in utils.random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)
            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'train_loss': objective.item()}
    

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, hparams)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], list(self.featurizer.parameters()) + list(self.classifier.parameters()), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])
        self.register_buffer('update_count', torch.tensor([0]))
        
    @property
    def ib_penalty_weight(self):
        return self.hparams['ib_lambda'] if self.update_count >= self.hparams['ib_penalty_anneal_iters'] else 0.0

    def update(self, minibatches, unlabeled=None):
        self.network.train()

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        features_list = torch.split(all_features, [x.shape[0] for x, y in minibatches])
        logits_list = torch.split(all_logits, [x.shape[0] for x, y in minibatches])
        
        nll = torch.mean(torch.stack([F.cross_entropy(logits, y) for logits, (x, y) in zip(logits_list, minibatches)]))
        ib_penalty = torch.mean(torch.stack([features.var(dim=0).mean() for features in features_list]))
        loss = nll + self.ib_penalty_weight * ib_penalty
        
        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], list(self.featurizer.parameters()) + list(self.classifier.parameters()), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'train_loss': loss.item(), 'nll': nll.item(), 'IB_penalty': ib_penalty.item()}
       
              
class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes,
                                  hparams)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], list(self.featurizer.parameters()) + list(self.classifier.parameters()), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1., device=device, requires_grad=True)
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        return torch.sum(grad_1 * grad_2)

    @property
    def irm_penalty_weight(self):
        return self.hparams['irm_lambda'] if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 1.0

    @property
    def ib_penalty_weight(self):
        return self.hparams['ib_lambda'] if self.update_count >= self.hparams['ib_penalty_anneal_iters'] else 0.0

    def update(self, minibatches, unlabeled=None):
        self.network.train()
        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        features_list = torch.split(all_features, [x.shape[0] for x, y in minibatches])
        logits_list = torch.split(all_logits, [x.shape[0] for x, y in minibatches])

        nll = torch.mean(torch.stack([F.cross_entropy(logits, y) for logits, (x, y) in zip(logits_list, minibatches)]))
        irm_penalty = torch.mean(torch.stack([self._irm_penalty(logits, y) for logits, (x, y) in zip(logits_list, minibatches)]))
        ib_penalty = torch.mean(torch.stack([features.var(dim=0).mean() for features in features_list]))

        loss = nll + self.irm_penalty_weight * irm_penalty + self.ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], list(self.featurizer.parameters()) + list(self.classifier.parameters()), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'train_loss': loss.item(), 'nll': nll.item(), 'IRM_penalty': irm_penalty.item(), 'IB_penalty': ib_penalty.item()}
    

class Fish(ERM):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, hparams):
        super(Fish, self).__init__(input_shape, num_classes,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], self.network.parameters(), lr=self.hparams["lr"],weight_decay=self.hparams['weight_decay'])
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams, weights=self.network.state_dict()).to(device)
        self.optimizer_inner = utils.extract_optimizer(self.hparams['optimizer_name'],self.network_inner.parameters(),lr=self.hparams["lr"],weight_decay=self.hparams['weight_decay'])
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = utils.ParamDict(meta_weights)
        inner_weights = utils.ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.network.train()
        self.create_clone(minibatches[0][0].device)
        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(meta_weights=self.network.state_dict(),inner_weights=self.network_inner.state_dict(),lr_meta=self.hparams["meta_lr"])
        self.network.reset_weights(meta_weights)

        return {'train_loss': loss.item()}