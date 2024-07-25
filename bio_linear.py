import torch
import torch.nn as nn
from torch.autograd import Function

class LinearFAFunction(Function):
    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_fa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, None, grad_bias


class LinearKPFunction(LinearFAFunction):
    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input, grad_weight, _, grad_bias = LinearFAFunction.backward(context, grad_output)

        # Update weight_fa using a decay factor, assuming this is done outside the backward pass
        # grad_weight_fa = (1 - decay_factor) * weight_fa + decay_factor * grad_weight

        return grad_input, grad_weight, None, grad_bias


class FALinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(FALinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features)) if bias else None

        self.weight_fa = nn.Parameter(torch.rand(output_features, input_features), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.weight_fa, a=5**0.5)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)


class KPLinear(FALinear):
    def forward(self, input):
        return LinearKPFunction.apply(input, self.weight, self.weight_fa, self.bias)

    def update_weight_fa(self, decay_factor=0.10):
        with torch.no_grad():
            self.weight_fa.copy_((1 - decay_factor) * self.weight_fa + decay_factor * self.weight.grad)



# # Code from https://github.com/limberc/DL-without-Weight-Transport-PyTorch/blob/master/linear.py
# # Contains the linear layer that uses Feedback Alignment and Kolen-Polack
#
# import torch
# import torch.nn as nn
# from torch.autograd import Function
#
# class LinearFAFunction(Function):
#     @staticmethod
#     # same as reference linear function, but with additional fa tensor for backward
#     def forward(context, input, weight, weight_fa, bias=None):
#         context.save_for_backward(input, weight, weight_fa, bias)
#         output = input.matmul(weight.t())
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output
#
#     @staticmethod
#     def backward(context, grad_output):
#         input, weight, weight_fa, bias = context.saved_variables
#         grad_input = grad_weight = grad_weight_fa = grad_bias = None
#         if context.needs_input_grad[0]:
#             # all of the logic of FA resides in this one line
#             # calculate the gradient of input with fixed fa tensor,
#             # rather than the "correct" model weight
#             grad_input = grad_output.matmul(weight_fa)
#         if context.needs_input_grad[1]:
#             # grad for weight with FA'ed grad_output from downstream layer
#             # it is same with original linear function
#             grad_weight = grad_output.t().matmul(input)
#         if bias is not None and context.needs_input_grad[3]:
#             grad_bias = grad_output.sum(0).squeeze(0)
#
#         return grad_input, grad_weight, grad_weight_fa, grad_bias
#
#
# class LinearKPFunction(LinearFAFunction):
#     @staticmethod
#     def backward(context, grad_output):
#         input, weight, weight_fa, bias = context.saved_variables
#         grad_input, grad_weight, grad_weight_fa, grad_bias = LinearFAFunction.backward(context, grad_output)
#         # Update the backward matrices of the Kolen-Pollack algorithm
#         grad_weight_fa = grad_weight
#
#         # decay_factor = 0.10
#         # grad_weight_fa = (1 - decay_factor) * weight_fa + decay_factor * grad_weight
#
#         return grad_input, grad_weight, grad_weight_fa, grad_bias
#
#
# class FALinear(nn.Module):
#     def __init__(self, input_features, output_features, bias=True):
#         super(FALinear, self).__init__()
#         self.input_features = input_features
#         self.output_features = output_features
#
#         # weight and bias for forward pass
#         # weight has transposed form; more efficient (so i heard) (transposed at forward pass)
#         self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(output_features))
#         else:
#             self.register_parameter('bias', None)
#
#         # fixed random weight and bias for FA backward pass
#         # does not need gradient
#         self.weight_fa = nn.Parameter(torch.rand(output_features, input_features,
#                                                  requires_grad=False).to(self.weight.device))
#         # weight initialization
#         torch.nn.init.kaiming_uniform_(self.weight)
#         torch.nn.init.kaiming_uniform_(self.weight_fa)
#         torch.nn.init.constant_(self.bias, 1)
#
#     def forward(self, input):
#         return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)
#
#
# class KPLinear(FALinear):
#     def __init__(self, input_features, output_features, bias=True):
#         super().__init__(input_features, output_features, bias)
#
#     def forward(self, input):
#         return LinearKPFunction.apply(input, self.weight, self.weight_fa, self.bias)
