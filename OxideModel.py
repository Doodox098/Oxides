from torch.autograd import Function
from torch.nn import Module
from scipy.special import expi
from torch.nn.parameter import Parameter
import torch
import numpy as np

class ExponentialIntegral(Function):
    """
    Implementation of exponential integral function Ei(x) = Integrate_{-∞}^x e^t/t dt
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return expi(input.detach().cpu())

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * torch.exp(input) / input


class ReLU2(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        return torch.relu(input) ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * torch.relu(input)


class OxideModel(torch.nn.Module):

    def __init__(
            self,
            e_state,
            v_state,
            t_state,
            b_state,
            config: dict = None,
            model: str = 'second',
    ):
        super().__init__()
        self.saved = False
        self.saved_old = False
        self.model = model
        self.e_state = e_state
        self.t_state = t_state
        self.v_state = v_state
        self.b_state = b_state
        self.config = config

        parameter_config = dict(dtype=torch.float64, requires_grad=True)
        self.E = Parameter(
            torch.tensor([np.log(e_state['init'] / e_state['scale'])], **parameter_config))
        self.T_max_delta = Parameter(torch.tensor([0], **parameter_config))
        self.T_beg_delta = Parameter(torch.tensor([0], **parameter_config))
        self.V_max = Parameter(
            torch.tensor([np.log(v_state['init'] / v_state['scale'])], **parameter_config))

    def get_t_beg(self):
        return self.b_state['init'] + self.b_state['delta'] * (
                    2 * torch.nn.functional.sigmoid(self.T_beg_delta * self.b_state['delta_scale']) - 1) \
            + self.t_state['delta'] * (
                        2 * torch.nn.functional.sigmoid(self.T_max_delta * self.t_state['delta_scale']) - 1)

    def get_t_max(self):
        return self.t_state['init'] + self.t_state['delta'] * (
                    2 * torch.nn.functional.sigmoid(self.T_max_delta * self.t_state['delta_scale']) - 1)

    def get_v_max(self):
        return torch.exp(self.V_max) * self.v_state['scale']

    def get_E(self):
        return torch.exp(self.E) * self.e_state['scale']

    def forward(self, input, global_shift):
        Ei = ExponentialIntegral.apply
        relu2 = ReLU2.apply

        def f(t, K, E):
            return K - E / t

        def integral(t, K, E):
            return torch.exp(K) * (t * torch.exp(-E / t) + E * Ei(-E / t))

        assert not torch.any(torch.isnan(self.E)), "NaN in s.E"
        assert not torch.any(torch.isnan(self.V_max)), f"NaN in s.V"
        assert not torch.any(torch.isnan(self.T_max_delta)), "NaN in T_max_delta"

        E = torch.exp(self.E) * self.e_state['scale']
        V = torch.exp(self.V_max) * self.v_state['scale']
        T_max = self.t_state['init'] + self.t_state['delta'] * (
                    2 * torch.nn.functional.sigmoid(self.T_max_delta * self.t_state['delta_scale']) - 1) + global_shift
        assert not torch.any(torch.isnan(E)), "NaN in E"
        assert not torch.any(torch.isnan(V)), "NaN in V"
        assert not torch.any(torch.isnan(T_max)), "NaN in T_max"

        if self.model == 'first':
            K = E / T_max + torch.log(E / (T_max ** 2))
            result = integral(input, K, E) - integral(T_max, K, E)
            assert not torch.any(torch.isnan(K)), f"NaN in K : {T_max =} {E =}"
            assert not torch.any(torch.isnan(integral(input, K, E))), f"NaN in integral({input=}, {K=}, {E=})"
            assert not torch.any(torch.isnan(integral(T_max, K, E))), f"NaN in integral({T_max=}, {K=}, {E=})"
            assert not torch.any(torch.isnan(V * torch.exp(f(input, K, E) - f(T_max, K, E) - result))), \
                "NaN in V * torch.exp(f(input, K ,E) - f(T_max, K, E) - result)"
            return V * torch.exp(f(input, K, E) - f(T_max, K, E) - result)

        if self.model == 'second':
            U = V ** 0.5
            K = E / T_max + (2 / 3) * torch.log((3 / 2) * E * U / (T_max ** 2))
            result = integral(input, K, E) - integral(T_max, K, E)
            return torch.exp(f(input, K, E) - f(T_max, K, E)) * relu2(
                U - (1 / 3) * torch.exp(f(T_max, K, E) / 2) * result)
