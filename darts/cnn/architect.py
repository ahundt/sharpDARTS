import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, momentum=0.9, weight_decay=3e-4, arch_learning_rate=3e-4, arch_betas=None, arch_weight_decay=1e-3):
        if arch_betas is None:
            arch_betas = (0.5, 0.999)
        self.network_momentum = momentum
        self.network_weight_decay = weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=arch_learning_rate,
            betas=arch_betas,
            weight_decay=arch_weight_decay)

    def _compute_unrolled_model(self, input_batch, target, eta, network_optimizer):
        loss = self.model._loss(input_batch, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(
                                 self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(
            loss,
            self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta))
        return unrolled_model

    def _compute_model(self, input_batch, target, eta, network_optimizer):
        moments = {}
        loss = self.model._loss(input_batch, target)
        for v in self.model.parameters():
            try:
                moments[v] = network_optimizer.state[v]['momentum_buffer'].mul_(
                    self.network_momentum)
            except:
                moments[v] = torch.zeros_like(v)
        dvs = torch.autograd.grad(loss, self.model.parameters())

        model_new = self.model.new()
        model_dict = self.model.state_dict()
        params = {}
        for dv, (k, v) in zip(dvs, self.model.named_parameters()):
            dv -= self.network_weight_decay * v
            v_ = torch.zeros_like(v)
            v_.data.copy_(v.data)
            v_.data.sub_(eta * (moments[v] + dv))
            params[k] = v_.view(v_.size())

        model_dict.update(params)
        model_new.load_state_dict(model_dict)

        return model_new.cuda()

    def step(self, input_train, target_train, input_valid, target_valid, eta,
             network_optimizer, unrolled=True):
        self.optimizer.zero_grad()
        if unrolled:
            loss = self._backward_step_unrolled(input_train, target_train, input_valid,
                                                target_valid, eta, network_optimizer)
        else:
            loss = self._backward_step(input_train, target_train, input_valid,
                                       target_valid, eta, network_optimizer)
        self.optimizer.step()
        # return loss for importance analysis of hyperparameter alphas
        return loss

    def _backward_step(self, input_train, target_train, input_valid,
                       target_valid, eta, network_optimizer):
        model = self._compute_model(input_train, target_train, eta,
                                    network_optimizer)
        loss = model._loss(input_valid, target_valid)
        loss.backward()
        dalpha = [v.grad for v in model.arch_parameters()]
        vector = [v.grad.data for v in model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train,
                                                      target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        # return loss for importance analysis of hyperparameter alphas
        return loss

    def _backward_step_unrolled(self, input_train, target_train, input_valid,
                                target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train,
                                                      eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train,
                                                      target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        # return loss for importance analysis of hyperparameter alphas
        return unrolled_loss

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset:offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input_batch, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input_batch, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input_batch, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
