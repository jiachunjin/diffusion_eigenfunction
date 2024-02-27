import math
import torch
import numpy as np
from functools import partial


class Erf(torch.nn.Module):
	def __init__(self):
		super(Erf, self).__init__()

	def forward(self, x):
		return x.erf()

class SinAndCos(torch.nn.Module):
	def __init__(self):
		super(SinAndCos, self).__init__()

	def forward(self, x):
		assert x.shape[1] % 2 == 0
		x1, x2 = x.chunk(2, dim=1)
		return torch.cat([torch.sin(x1), torch.cos(x2)], 1)

class ParallelLinear(torch.nn.Module):
	def __init__(self, in_features, out_features, num_copies):
		super(ParallelLinear, self).__init__()
		self.register_parameter('weight', torch.nn.Parameter(torch.randn(num_copies, out_features, in_features)))
		self.register_parameter('bias', torch.nn.Parameter(torch.zeros(num_copies, out_features, 1)))

		for i in range(num_copies):
			torch.nn.init.normal_(self.weight[i], 0, math.sqrt(2./in_features))
		torch.nn.init.zeros_(self.bias)

	def forward(self, x):
		if x.dim() == 2:
			return torch.tensordot(self.weight, x, [[2], [1]]) + self.bias
		else:
			return self.weight @ x + self.bias

class ParallelMLP(torch.nn.Module):
	def __init__(self, in_features, out_features, num_copies, num_layers, hidden_size=64, nonlinearity='relu'):
		super(ParallelMLP, self).__init__()

		if nonlinearity == 'relu':
			nonlinearity=torch.nn.ReLU
		elif 'lrelu' in nonlinearity:
			nonlinearity=partial(torch.nn.LeakyReLU, float(nonlinearity.replace("lrelu", "")))
		elif nonlinearity == 'erf':
			nonlinearity=Erf
		elif nonlinearity == 'sin_and_cos':
			nonlinearity=SinAndCos
		else:
			raise NotImplementedError

		if num_layers == 1:
			self.fn = torch.nn.Sequential(
				ParallelLinear(in_features, out_features, num_copies))
		else:
			layers = [ParallelLinear(in_features, hidden_size, num_copies),
					  nonlinearity(),
					  ParallelLinear(hidden_size, out_features, num_copies)]
			for _ in range(num_layers - 2):
				layers.insert(2, nonlinearity())
				layers.insert(2, ParallelLinear(hidden_size, hidden_size, num_copies))
			self.fn = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.fn(x).permute(2, 1, 0)

class NeuralEigenFunctions(torch.nn.Module):
	def __init__(self, k, nonlinearity='relu', input_size=1,
				 hidden_size=64, num_layers=3, output_size=1, momentum=0.9,
				 normalize_over=[0]):
		super(NeuralEigenFunctions, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.fn = ParallelMLP(input_size, output_size, k, num_layers, hidden_size, nonlinearity)
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		ret_raw = self.fn(x).squeeze()
		if self.training:
			norm_ = ret_raw.norm(dim=self.normalize_over) / math.sqrt(
				np.prod([ret_raw.shape[dim] for dim in self.normalize_over]))
			with torch.no_grad():
				if self.num_calls == 0:
					self.eigennorm.copy_(norm_.data)
				else:
					self.eigennorm.mul_(self.momentum).add_(
						norm_.data, alpha = 1-self.momentum)
				self.num_calls += 1
		else:
			norm_ = self.eigennorm
		return ret_raw / norm_