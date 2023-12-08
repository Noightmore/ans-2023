import functools
from typing import Any, Optional, Union

import torch

from ans.autograd import Variable


class Function:

    @classmethod
    def apply(cls, *inputs: Union[Variable, Any], **params: Any) -> Variable:
        tensor_args = [i.data if isinstance(i, Variable) else i for i in inputs]
        output_data, cache = cls.forward(*tensor_args, **params)
        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, ...]:
            dinputs = cls.backward(dout, cache=cache)
            return tuple(dinputs[i] for i, inp in enumerate(inputs) if isinstance(inp, Variable))
        grad_fn.name = f"{cls.__name__}.backward"
        return Variable(
            output_data,
            parents=tuple(i for i in inputs if isinstance(i, Variable)),
            grad_fn=grad_fn
        )

    @staticmethod
    def forward(*inputs: torch.Tensor, **params: Any) -> tuple[torch.Tensor, tuple]:
        raise NotImplementedError

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return str(self)


class Linear(Function):

    @staticmethod
    def forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_features)
            weight: shape (num_features, num_out)
            bias: shape (num_out,)
        Returns:
            output: shape (num_samples, num_out)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        output = torch.mm(input, weight) + bias
        cache = (input, weight)
        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_out)
            cache: cache from the forward pass
        Returns:
            tuple of gradient w.r.t. input, weight, bias in this order
        """

        ########################################
        # TODO: implement

        input, weight = cache
        dweight = torch.mm(input.t(), doutput)
        dbias = torch.sum(doutput, axis=0)
        dinput = torch.mm(doutput, weight.t())

        # ENDTODO
        ########################################

        return dinput, dweight, dbias


class Sigmoid(Function):

    @staticmethod
    def forward(input: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_features)
        Returns:
            output: shape (num_samples, num_features)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        output = 1 / (1 + torch.exp(-input))
        cache = (output,)

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_out)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        input = cache[0]
        dinput = doutput * input * (1 - input)

        # ENDTODO
        ########################################

        return dinput,


class SoftmaxCrossEntropy(Function):

    @staticmethod
    def forward(scores: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            scores: shape (num_samples, num_out)
            targets: shape (num_samples,); dtype torch.int64
        Returns:
            output: shape () (scalar tensor)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        maxs, _ = torch.max(scores, dim=1)
        scores = scores - maxs.unsqueeze(1)
        scoresOH = scores[torch.arange(scores.shape[0]), targets]
        output = torch.mean(-scoresOH + torch.logsumexp(scores, dim=1))
        cache = (scores, targets)

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape ()
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. scores (single-element tuple)
        """

        ########################################
        # TODO: implement

        scores, targets = cache
        exp_scores = torch.exp(scores)
        dscores = exp_scores / torch.sum(exp_scores, dim=1).unsqueeze(1)
        dscores[torch.arange(targets.shape[0]), targets] -= 1
        dscores = dscores * doutput / scores.shape[0]

        # ENDTODO
        ########################################

        return dscores,


class ReLU(Function):

    @staticmethod
    def forward(input: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, ...)
        Returns:
            output: shape (num_samples, ...)
            cache: tuple of intermediate results to use in backward

        Operation is not inplace, i.e. it does not modify the input.
        """

        ########################################
        # TODO: implement

        output = torch.where(input > 0, input, 0)
        cache = output,

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        input = cache[0]
        dinput = torch.where(input > 0, doutput, 0)

        # ENDTODO
        ########################################

        return dinput,


class Tanh(Function):

    @staticmethod
    def forward(input: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, ...)
        Returns:
            output: shape (num_samples, ...)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return dinput,


class Dropout(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            p: float = 0.5,
            training: bool = False,
            seed: Optional[int] = None
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, ...)
            p: probability of element being zeroed out
            training: whether in training mode or eval mode
            seed: enable deterministic behavior (useful for gradient check)
        Returns:
            output: shape (num_samples, ...)
            cache: tuple of intermediate results to use in backward
        """

        # deterministic behavior for gradient check
        if seed is not None:
            torch.manual_seed(seed)

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return dinput,


class BatchNorm1d(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            gamma: Optional[torch.Tensor],
            beta: Optional[torch.Tensor],
            running_mean: Optional[torch.Tensor] = None,
            running_var: Optional[torch.Tensor] = None,
            momentum: float = 0.9,
            eps: float = 1e-05,
            training: bool = False
    ) -> tuple[torch.Tensor, tuple]:
        """

        Args:
            input: shape (num_samples, num_features)
            gamma: shape (num_features,)
            beta: shape (num_features,)
            running_mean: shape (num_features,)
            running_var: shape (num_features,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_features)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_features)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return dinput, dgamma, dbeta


class BatchNorm2d(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            gamma: Optional[torch.Tensor],
            beta: Optional[torch.Tensor],
            running_mean: Optional[torch.Tensor] = None,
            running_var: Optional[torch.Tensor] = None,
            momentum: float = 0.9,
            eps: float = 1e-05,
            training: bool = False
    ) -> tuple[torch.Tensor, tuple]:
        """
        Spatial BatchNorm for convolutional networks

        Args:
            input: shape (num_samples, num_channels, height, width)
            gamma: shape (num_channels,)
            beta: shape (num_channels,)
            running_mean: shape (num_channels,)
            running_var: shape (num_channels,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_channels, height, width)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height, width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return dinput, dgamma, dbeta


class Conv2d(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_channels, height, width)
            weight: shape (num_filters, num_channels, kernel_size[0], kernel_size[1])
            bias: shape (num_filters,)
            stride: convolution step size
            padding: how much should the input be padded on each side by zeroes
            dilation: see torch.nn.functional.conv2d
            groups: see torch.nn.functional.conv2d

        Returns:
            output: shape (num_samples, num_filters, output_height, output_width)
            cache: tuple of intermediate results to use in backward
        """
        ########################################
        # TODO: implement

        output = torch.nn.functional.conv2d(input=input,
                                            weight=weight,
                                            bias=bias,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups)
        cache = (input, output, bias, weight, stride, padding, dilation, groups)

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_filters, output_height, output_width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input, weight and bias
        """

        ########################################
        # TODO: implement

        input, output, bias, weight, stride, padding, dilation, groups = cache
        output_padding = 0

        if stride > 1:
            o_pad_h = int(input.shape[2] - (doutput.shape[2] - 1) * stride + 2 * padding - (weight.shape[2] - 1) * dilation - 1)
            o_pad_w = int(input.shape[3] - (doutput.shape[3] - 1) * stride + 2 * padding - (weight.shape[3] - 1) * dilation - 1)
            output_padding = (o_pad_h, o_pad_w)
        dinput = torch.nn.functional.conv_transpose2d(input=doutput,
                                                      weight=weight,
                                                      stride=stride,
                                                      padding=padding,
                                                      dilation=dilation,
                                                      output_padding=output_padding,
                                                      groups=groups)

        dbias = doutput.sum(dim=[0, 2, 3])
        dweight = torch.nn.functional.conv2d(input=input.transpose(0, 1),
                                             weight=doutput.transpose(0, 1),
                                             stride=dilation,
                                             padding=padding,
                                             dilation=stride,
                                             groups=groups).transpose(0, 1)
        k = weight.shape[2]
        dweight = dweight[:, :, 0:k, 0:k]

        # ENDTODO
        ########################################

        return dinput, dweight, dbias


class MaxPool2d(Function):

    @staticmethod
    def forward(input: torch.Tensor, window_size: int = 2) -> tuple[torch.Tensor, tuple]:
        """

        Args:
            input: shape (num_samples, num_channels, height, width)
            window_size: size of pooling window
        Returns:
            output: shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        num_samples, num_channels, height, width = input.shape
        input = input[:, :, 0:(height - height % window_size), :]
        input = input[:, :, :, 0:(width - width % window_size)]

        output = torch.reshape(input, (num_samples, num_channels,
                                       input.shape[2]//window_size,
                                       window_size, input.shape[3]//window_size, window_size))

        output = output.transpose(3, 4)
        output = torch.reshape(output, (num_samples, num_channels,
                                        input.shape[2]//window_size,
                                        input.shape[3]//window_size, window_size**2))

        output, idx = torch.max(output, dim=4)
        cache = ((num_samples, num_channels, height, width), window_size, idx, output)

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        input_shape, window_size, idx, output = cache

        num_samples, num_channels, height, width = input_shape

        dinput = torch.zeros(num_samples, num_channels, height//window_size,
                             width//window_size, window_size ** 2,
                             dtype=doutput.dtype).to(device=doutput.device)

        dinput = dinput.scatter_(-1, idx.unsqueeze(-1), 1)
        dinput[dinput == 1] = doutput.flatten()
        dinput = torch.reshape(dinput, (num_samples, num_channels, height//window_size,
                                        width//window_size, window_size, window_size))

        dinput = dinput.transpose(3, 4)
        dinput = torch.reshape(dinput, (num_samples, num_channels,
                                        height - height % window_size, width - width%window_size))

        dinput = torch.nn.functional.pad(dinput, (0, width-dinput.shape[3], 0, height-dinput.shape[2]))

        # ENDTODO
        ########################################

        return dinput,
