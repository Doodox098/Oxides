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
        # Добавляем ограничения на входные значения
        # input_clamped = torch.clamp(input, min=-100, max=100)
        input_clamped = input
        ctx.save_for_backward(input_clamped)
        with torch.no_grad():
            output = torch.from_numpy(expi(input_clamped.detach().cpu().numpy()))
        return output.to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Добавляем стабилизацию для малых значений
        safe_input = torch.where(torch.abs(input) < 1e-10,
                                 torch.ones_like(input) * 1e-10,
                                 input)
        return grad_output * torch.exp(safe_input) / safe_input


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
        return self.b_state['init'] + self.b_state['delta'] \
                * torch.nn.functional.tanh(self.T_beg_delta * self.b_state['delta_scale'])
    def get_t_max(self):
        return self.t_state['init'] + self.t_state['delta'] \
                * torch.nn.functional.tanh(self.T_max_delta * self.t_state['delta_scale'])

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

        # Добавляем проверки на NaN
        assert not torch.any(torch.isnan(self.E)), "NaN in self.E"
        assert not torch.any(torch.isnan(self.V_max)), "NaN in self.V_max"
        assert not torch.any(torch.isnan(self.T_max_delta)), "NaN in T_max_delta"

        E = torch.clamp(torch.exp(self.E) * self.e_state['scale'], min=1e-10, max=1e10)
        V = torch.clamp(torch.exp(self.V_max) * self.v_state['scale'], min=1e-10, max=1e10)
        T_max = self.t_state['init'] + self.t_state['delta'] \
                * torch.nn.functional.tanh(self.T_max_delta * self.t_state['delta_scale']) + global_shift

        # Стабилизируем вычисления
        safe_T_max = torch.clamp(T_max, min=1e-10)
        safe_input = torch.clamp(input, min=1e-10)

        if self.model == 'first':
            K = E / safe_T_max + torch.log(E / (safe_T_max ** 2))
            # Стабилизируем вычисления интегралов
            integral_input = integral(safe_input, K, E)
            integral_T_max = integral(safe_T_max, K, E)
            result = integral_input - integral_T_max

            # Стабилизируем экспоненту
            exp_arg = f(safe_input, K, E) - f(safe_T_max, K, E) - result
            exp_arg = torch.clamp(exp_arg, min=-100, max=100)
            return V * torch.exp(exp_arg)

        if self.model == 'second':
            U = torch.sqrt(V)
            K = E / safe_T_max + (2 / 3) * torch.log((3 / 2) * E * U / (safe_T_max ** 2))
            # Стабилизируем вычисления интегралов
            integral_input = integral(safe_input, K, E)
            integral_T_max = integral(safe_T_max, K, E)
            result = integral_input - integral_T_max

            # Стабилизируем экспоненту
            exp_arg = f(safe_input, K, E) - f(safe_T_max, K, E)
            exp_arg = torch.clamp(exp_arg, min=-100, max=100)
            return torch.exp(exp_arg) * relu2(
                U - (1 / 3) * torch.exp(f(safe_T_max, K, E) / 2) * result)
