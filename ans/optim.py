import torch

from ans.autograd import Variable


class Optimizer:

    def __init__(self, parameters: list[Variable]) -> None:
        self.parameters = parameters

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None


class SGD(Optimizer):

    def __init__(
            self,
            parameters: list[Variable],
            learning_rate: float = 1e-3,
            momentum: float = 0.,
            weight_decay: float = 0.
    ) -> None:
        super().__init__(parameters)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        ########################################
        # TODO: init _velocities to zeros

        #raise NotImplementedError

        # # move device to GPU if available
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #
        # # move device to GPU if available
        # for parameter in self.parameters:
        #     if parameter.grad is None:
        #         continue
        #
        #     parameter.data = parameter.data.to(device)
        #     parameter.grad = parameter.grad.to(device)

        self._velocities: dict[Variable, torch.Tensor] = dict.fromkeys(parameters, torch.tensor(0, dtype=torch.float16))

        # ENDTODO
        ########################################

    def step(self) -> None:
        ########################################
        # TODO: implement

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #
        # # move device to GPU if available
        # for parameter in self.parameters:
        #     parameter.data = parameter.data.to(device)
        #     parameter.grad = parameter.grad.to(device)

        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad + self.weight_decay * parameter.data
            self._velocities[parameter] = self.momentum * self._velocities[parameter] - self.learning_rate * grad
            parameter.data = parameter.data + self._velocities[parameter]

        # ENDTODO
        ########################################


class Adam(Optimizer):

    def __init__(
            self,
            parameters: list[Variable],
            learning_rate: float = 1e-3,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-08,
            weight_decay: float = 0.,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        ########################################
        # TODO: init _num_steps to zero, _m to zeros, _v to zeros

        self._num_steps = 0
        self._m: dict[Variable, torch.Tensor] = dict.fromkeys(parameters, torch.tensor(0, dtype=torch.float16))
        self._v: dict[Variable, torch.Tensor] = dict.fromkeys(parameters, torch.tensor(0, dtype=torch.float16))

        # ENDTODO
        ########################################

    def step(self) -> None:
        ########################################
        # TODO: implement

        self._num_steps += 1
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad + self.weight_decay * parameter.data
            self._m[parameter] = self.beta1 * self._m[parameter] + (1 - self.beta1) * grad
            self._v[parameter] = self.beta2 * self._v[parameter] + (1 - self.beta2) * (grad ** 2)
            m = self._m[parameter] / (1 - (self.beta1 ** self._num_steps))
            v = self._v[parameter] / (1 - (self.beta2 ** self._num_steps))
            parameter.data -= self.learning_rate * (m / (torch.sqrt(v) + self.eps))

        # ENDTODO
        ########################################
