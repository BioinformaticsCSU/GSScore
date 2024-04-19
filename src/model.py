
import sys, os, argparse, math
import traceback
import torch
from torch import nn, Tensor
import numpy as np
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq import utils
from fairseq.modules.quant_noise import quant_noise

from typing import List, Optional, Tuple, Callable


def init_params(module, n_layers):
	if isinstance(module, nn.Linear):
		module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
		if module.bias is not None:
			module.bias.data.zero_()
	if isinstance(module, nn.Embedding):
		module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
	def __init__(self, num_heads, num_atoms, hidden_dim, n_layers, num_in_degree=0, num_out_degree=0, bUseDegree=False):
		super(GraphNodeFeature, self).__init__()
		self.num_heads = num_heads
		self.num_atoms = num_atoms
		self.atom_encoder = nn.Linear(in_features=156, out_features=hidden_dim, bias=False)
		if bUseDegree:
			self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
			self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
		self.bUseDegree = bUseDegree
		self.graph_token = nn.Embedding(1, hidden_dim)
		self.apply(lambda module: init_params(module, n_layers=n_layers))

	def forward(self, batched_data):
		if self.bUseDegree:
			x, in_degree, out_degree = (
				batched_data["x"],
				batched_data["in_degree"],
				batched_data["out_degree"],
			)
		else:
			x = (batched_data["x"])
		# x = (batched_data["x"])
		n_graph, n_node = x.size()[:2]

		# node feauture + graph token
		MyDebug = self.atom_encoder(x)
		node_feature = MyDebug  # MyDebug.sum(dim=-2)  # [n_graph, n_node, n_hidden]

		# if self.flag and perturb is not None:
		#	 node_feature += perturb
		# print('in_degree == out_degree? (%s)' % ((in_degree==out_degree).all()))

		if self.bUseDegree:
			MyDebug1 = self.in_degree_encoder(in_degree)
			MyDebug2 = self.out_degree_encoder(out_degree)
			node_feature = (node_feature + MyDebug1 + MyDebug2)

		graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

		graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

		return graph_node_feature


class GraphAttnBias(nn.Module):
	"""
	Compute attention bias for each head.
	"""

	def __init__(
			self,
			num_heads,
			num_atoms,
			num_edges,
			num_spatial,
			num_edge_dis,
			hidden_dim,
			edge_type,
			multi_hop_max_dist,
			n_layers,
	):
		super(GraphAttnBias, self).__init__()
		self.num_heads = num_heads
		self.multi_hop_max_dist = multi_hop_max_dist
		# self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
		self.edge_encoder = nn.Linear(in_features=8, out_features=num_heads, bias=False)
		self.edge_type = edge_type
		# if self.edge_type == "multi_hop":
		# 	self.edge_dis_encoder = nn.Embedding(
		# 		num_edge_dis * num_heads * num_heads, 1
		# 	)
		self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
		# self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
		self.spatial_pos_encoder = nn.Linear(in_features=1, out_features=num_heads, bias=False)

		self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

		self.apply(lambda module: init_params(module, n_layers=n_layers))

	def forward(self, batched_data):
		attn_bias, spatial_pos, x = (
			batched_data["attn_bias"],
			batched_data["spatial_pos"],
			batched_data["x"],
		)
		# in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
		edge_input = batched_data["edge_input"]

		n_graph, n_node = x.size()[:2]
		graph_attn_bias = attn_bias.clone()
		graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
			1, self.num_heads, 1, 1
		)  # [n_graph, n_head, n_node+1, n_node+1]

		# Spatial Encoding
		spatial_pos_temp = spatial_pos.unsqueeze(-1)
		MyDebug = self.spatial_pos_encoder(spatial_pos_temp)
		spatial_pos_bias = MyDebug.permute(0, 3, 1, 2)
		graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

		# reset spatial pos here
		# output value a learnable scalar which will serve as a bias term in the self-attention module.
		t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
		graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
		graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

		# edge feature : multi-hop
		spatial_pos_ = spatial_pos.clone()
		spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
		# set 1 to 1, x > 1 to x - 1
		spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
		if self.multi_hop_max_dist > 0:
			spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
			edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
		# [n_graph, n_node, n_node, max_dist, n_head]
		MyDebug = self.edge_encoder(edge_input)
		edge_input = MyDebug
		max_dist = edge_input.size(-2)
		edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
			max_dist, -1, self.num_heads
		)
		MyDebug = self.edge_dis_encoder.weight.reshape(
			-1, self.num_heads, self.num_heads
		)
		edge_input_flat = torch.bmm(
			edge_input_flat,
			MyDebug[:max_dist, :, :],
		)
		edge_input = edge_input_flat.reshape(
			max_dist, n_graph, n_node, n_node, self.num_heads
		).permute(1, 2, 3, 0, 4)
		MyDebug1 = edge_input.sum(-2)
		MyDebug2 = spatial_pos_.float().unsqueeze(-1)
		edge_input = (
				MyDebug1 / MyDebug2
		).permute(0, 3, 1, 2)

		graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
		MyDebug3 = attn_bias.unsqueeze(1)
		graph_attn_bias = graph_attn_bias + MyDebug3  # reset

		return graph_attn_bias


import logging

logger = logging.getLogger(__name__)


class MultiheadAttention(nn.Module):
	"""Multi-headed attention.

	See "Attention Is All You Need" for more details.
	"""

	def __init__(
			self,
			embed_dim,
			num_heads,
			kdim=None,
			vdim=None,
			dropout=0.0,
			bias=True,
			self_attention=False,
			q_noise=0.0,
			qn_block_size=8,
	):
		super().__init__()
		self.embed_dim = embed_dim
		self.kdim = kdim if kdim is not None else embed_dim
		self.vdim = vdim if vdim is not None else embed_dim
		self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

		self.num_heads = num_heads
		self.dropout_module = FairseqDropout(
			dropout, module_name=self.__class__.__name__
		)

		self.head_dim = embed_dim // num_heads
		assert (
				self.head_dim * num_heads == self.embed_dim
		), "embed_dim must be divisible by num_heads"
		self.scaling = self.head_dim ** -0.5

		self.self_attention = self_attention

		assert self.self_attention, "Only support self attention"

		assert not self.self_attention or self.qkv_same_dim, (
			"Self-attention requires query, key and " "value to be of the same size"
		)

		self.k_proj = quant_noise(
			nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
		)
		self.v_proj = quant_noise(
			nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
		)
		self.q_proj = quant_noise(
			nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
		)

		self.out_proj = quant_noise(
			nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
		)

		self.reset_parameters()

		self.onnx_trace = False

	def prepare_for_onnx_export_(self):
		raise NotImplementedError

	def reset_parameters(self):
		if self.qkv_same_dim:
			# Empirically observed the convergence to be much better with
			# the scaled initialization
			nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
			nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
			nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
		else:
			nn.init.xavier_uniform_(self.k_proj.weight)
			nn.init.xavier_uniform_(self.v_proj.weight)
			nn.init.xavier_uniform_(self.q_proj.weight)

		nn.init.xavier_uniform_(self.out_proj.weight)
		if self.out_proj.bias is not None:
			nn.init.constant_(self.out_proj.bias, 0.0)

	def forward(
			self,
			query,
			key: Optional[Tensor],
			value: Optional[Tensor],
			attn_bias: Optional[Tensor],
			key_padding_mask: Optional[Tensor] = None,
			need_weights: bool = True,
			attn_mask: Optional[Tensor] = None,
			before_softmax: bool = False,
			need_head_weights: bool = False,
	) -> Tuple[Tensor, Optional[Tensor]]:
		"""Input shape: Time x Batch x Channel

		Args:
			key_padding_mask (ByteTensor, optional): mask to exclude
				keys that are pads, of shape `(batch, src_len)`, where
				padding elements are indicated by 1s.
			need_weights (bool, optional): return the attention weights,
				averaged over heads (default: False).
			attn_mask (ByteTensor, optional): typically used to
				implement causal attention, where the mask prevents the
				attention from looking forward in time (default: None).
			before_softmax (bool, optional): return the raw attention
				weights and values before the attention softmax.
			need_head_weights (bool, optional): return the attention
				weights for each head. Implies *need_weights*. Default:
				return the average attention weights over all heads.
		"""
		if need_head_weights:
			need_weights = True

		tgt_len, bsz, embed_dim = query.size()
		src_len = tgt_len
		assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
		assert list(query.size()) == [tgt_len, bsz, embed_dim]
		if key is not None:
			src_len, key_bsz, _ = key.size()
			if not torch.jit.is_scripting():
				assert key_bsz == bsz
				assert value is not None
				assert src_len, bsz == value.shape[:2]

		q = self.q_proj(query)
		k = self.k_proj(query)
		v = self.v_proj(query)
		q *= self.scaling

		q = (
			q.contiguous()
				.view(tgt_len, bsz * self.num_heads, self.head_dim)
				.transpose(0, 1)
		)
		if k is not None:
			k = (
				k.contiguous()
					.view(-1, bsz * self.num_heads, self.head_dim)
					.transpose(0, 1)
			)
		if v is not None:
			v = (
				v.contiguous()
					.view(-1, bsz * self.num_heads, self.head_dim)
					.transpose(0, 1)
			)

		assert k is not None
		assert k.size(1) == src_len

		# This is part of a workaround to get around fork/join parallelism
		# not supporting Optional types.
		if key_padding_mask is not None and key_padding_mask.dim() == 0:
			key_padding_mask = None

		if key_padding_mask is not None:
			assert key_padding_mask.size(0) == bsz
			assert key_padding_mask.size(1) == src_len
		attn_weights = torch.bmm(q, k.transpose(1, 2))
		attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

		assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

		if attn_bias is not None:
			MyDebug = attn_bias.view(bsz * self.num_heads, tgt_len, src_len)
			attn_weights += MyDebug

		if attn_mask is not None:
			attn_mask = attn_mask.unsqueeze(0)
			attn_weights += attn_mask

		if key_padding_mask is not None:
			# don't attend to padding symbols
			attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
			attn_weights = attn_weights.masked_fill(
				key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
				float("-inf"),
			)
			attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

		if before_softmax:
			return attn_weights, v

		attn_weights_float = utils.softmax(
			attn_weights, dim=-1, onnx_trace=self.onnx_trace
		)
		attn_weights = attn_weights_float.type_as(attn_weights)
		attn_probs = self.dropout_module(attn_weights)

		assert v is not None
		attn = torch.bmm(attn_probs, v)
		assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

		attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
		attn = self.out_proj(attn)

		attn_weights: Optional[Tensor] = None
		if need_weights:
			attn_weights = attn_weights_float.view(
				bsz, self.num_heads, tgt_len, src_len
			).transpose(1, 0)
			if not need_head_weights:
				# average attention weights over heads
				attn_weights = attn_weights.mean(dim=0)

		return attn, attn_weights

	def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
		return attn_weights

	def upgrade_state_dict_named(self, state_dict, name):
		prefix = name + "." if name != "" else ""
		items_to_add = {}
		keys_to_remove = []
		for k in state_dict.keys():
			if k.endswith(prefix + "in_proj_weight"):
				# in_proj_weight used to be q + k + v with same dimensions
				dim = int(state_dict[k].shape[0] / 3)
				items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
				items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
				items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

				keys_to_remove.append(k)

				k_bias = prefix + "in_proj_bias"
				if k_bias in state_dict.keys():
					dim = int(state_dict[k].shape[0] / 3)
					items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
					items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
														   dim: 2 * dim
														   ]
					items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

					keys_to_remove.append(prefix + "in_proj_bias")

		for k in keys_to_remove:
			del state_dict[k]

		for key, value in items_to_add.items():
			state_dict[key] = value


def init_graphormer_params(module):
	"""
	Initialize the weights specific to the Graphormer Model.
	"""

	def normal_(data):
		# with FSDP, module params will be on CUDA, so we cast them back to CPU
		# so that the RNG is consistent with and without FSDP
		data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

	if isinstance(module, nn.Linear):
		normal_(module.weight.data)
		if module.bias is not None:
			module.bias.data.zero_()
	if isinstance(module, nn.Embedding):
		normal_(module.weight.data)
		if module.padding_idx is not None:
			module.weight.data[module.padding_idx].zero_()
	if isinstance(module, MultiheadAttention):
		normal_(module.q_proj.weight.data)
		normal_(module.k_proj.weight.data)
		normal_(module.v_proj.weight.data)


class GraphormerGraphEncoderLayer(nn.Module):
	def __init__(
			self,
			embedding_dim: int = 768,
			ffn_embedding_dim: int = 3072,
			num_attention_heads: int = 8,
			dropout: float = 0.1,
			attention_dropout: float = 0.1,
			activation_dropout: float = 0.1,
			activation_fn: str = "relu",
			export: bool = False,
			q_noise: float = 0.0,
			qn_block_size: int = 8,
			init_fn: Callable = None,
			pre_layernorm: bool = False,
	) -> None:
		super().__init__()

		if init_fn is not None:
			init_fn()

		# Initialize parameters
		self.embedding_dim = embedding_dim
		self.num_attention_heads = num_attention_heads
		self.attention_dropout = attention_dropout
		self.q_noise = q_noise
		self.qn_block_size = qn_block_size
		self.pre_layernorm = pre_layernorm

		self.dropout_module = FairseqDropout(
			dropout, module_name=self.__class__.__name__
		)
		self.activation_dropout_module = FairseqDropout(
			activation_dropout, module_name=self.__class__.__name__
		)

		# Initialize blocks
		self.activation_fn = utils.get_activation_fn(activation_fn)
		self.self_attn = self.build_self_attention(
			self.embedding_dim,
			num_attention_heads,
			dropout=attention_dropout,
			self_attention=True,
			q_noise=q_noise,
			qn_block_size=qn_block_size,
		)

		# layer norm associated with the self attention layer
		self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

		self.fc1 = self.build_fc1(
			self.embedding_dim,
			ffn_embedding_dim,
			q_noise=q_noise,
			qn_block_size=qn_block_size,
		)
		self.fc2 = self.build_fc2(
			ffn_embedding_dim,
			self.embedding_dim,
			q_noise=q_noise,
			qn_block_size=qn_block_size,
		)

		# layer norm associated with the position wise feed-forward NN
		self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

	def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
		return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

	def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
		return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

	def build_self_attention(
			self,
			embed_dim,
			num_attention_heads,
			dropout,
			self_attention,
			q_noise,
			qn_block_size,
	):
		return MultiheadAttention(
			embed_dim,
			num_attention_heads,
			dropout=dropout,
			self_attention=True,
			q_noise=q_noise,
			qn_block_size=qn_block_size,
		)

	def forward(
			self,
			x: torch.Tensor,
			self_attn_bias: Optional[torch.Tensor] = None,
			self_attn_mask: Optional[torch.Tensor] = None,
			self_attn_padding_mask: Optional[torch.Tensor] = None,
	):
		"""
		LayerNorm is applied either before or after the self-attention/ffn
		modules similar to the original Transformer implementation.
		"""
		# x: T x B x C
		residual = x
		if self.pre_layernorm:
			x = self.self_attn_layer_norm(x)
		x, attn = self.self_attn(
			query=x,
			key=x,
			value=x,
			attn_bias=self_attn_bias,
			key_padding_mask=self_attn_padding_mask,
			need_weights=False,
			attn_mask=self_attn_mask,
		)
		x = self.dropout_module(x)
		x = residual + x
		if not self.pre_layernorm:
			x = self.self_attn_layer_norm(x)

		residual = x
		if self.pre_layernorm:
			x = self.final_layer_norm(x)
		x = self.activation_fn(self.fc1(x))
		x = self.activation_dropout_module(x)
		x = self.fc2(x)
		x = self.dropout_module(x)
		x = residual + x
		if not self.pre_layernorm:
			x = self.final_layer_norm(x)
		return x, attn


class GraphormerGraphEncoder(nn.Module):
	def __init__(
			self,
			num_atoms: int,
			num_in_degree: int,
			num_out_degree: int,
			num_edges: int,
			num_spatial: int,
			num_edge_dis: int,
			edge_type: str,
			multi_hop_max_dist: int,
			num_encoder_layers: int = 12,
			embedding_dim: int = 768,
			ffn_embedding_dim: int = 768,
			num_attention_heads: int = 32,
			dropout: float = 0.1,
			attention_dropout: float = 0.1,
			activation_dropout: float = 0.1,
			layerdrop: float = 0.0,
			encoder_normalize_before: bool = False,
			pre_layernorm: bool = False,
			apply_graphormer_init: bool = False,
			activation_fn: str = "gelu",
			embed_scale: float = None,
			freeze_embeddings: bool = False,
			n_trans_layers_to_freeze: int = 0,
			export: bool = False,
			traceable: bool = False,
			q_noise: float = 0.0,
			qn_block_size: int = 8,
			bUseDegree: bool = False,
	) -> None:

		super().__init__()
		self.dropout_module = FairseqDropout(
			dropout, module_name=self.__class__.__name__
		)
		self.layerdrop = layerdrop
		self.embedding_dim = embedding_dim
		self.apply_graphormer_init = apply_graphormer_init
		self.traceable = traceable

		self.graph_node_feature = GraphNodeFeature(
			num_heads=num_attention_heads,
			num_atoms=num_atoms,
			num_in_degree=num_in_degree,
			num_out_degree=num_out_degree,
			hidden_dim=embedding_dim,
			n_layers=num_encoder_layers,
			bUseDegree=bUseDegree,
		)

		self.graph_attn_bias = GraphAttnBias(
			num_heads=num_attention_heads,
			num_atoms=num_atoms,
			num_edges=num_edges,
			num_spatial=num_spatial,
			num_edge_dis=num_edge_dis,
			edge_type=edge_type,
			multi_hop_max_dist=multi_hop_max_dist,
			hidden_dim=embedding_dim,
			n_layers=num_encoder_layers,
		)

		self.embed_scale = embed_scale

		if q_noise > 0:
			self.quant_noise = apply_quant_noise_(
				nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
				q_noise,
				qn_block_size,
			)
		else:
			self.quant_noise = None

		if encoder_normalize_before:
			self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
		else:
			self.emb_layer_norm = None

		if pre_layernorm:
			self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

		if self.layerdrop > 0.0:
			self.layers = LayerDropModuleList(p=self.layerdrop)
		else:
			self.layers = nn.ModuleList([])
		self.layers.extend(
			[
				self.build_graphormer_graph_encoder_layer(
					embedding_dim=self.embedding_dim,
					ffn_embedding_dim=ffn_embedding_dim,
					num_attention_heads=num_attention_heads,
					dropout=self.dropout_module.p,
					attention_dropout=attention_dropout,
					activation_dropout=activation_dropout,
					activation_fn=activation_fn,
					export=export,
					q_noise=q_noise,
					qn_block_size=qn_block_size,
					pre_layernorm=pre_layernorm,
				)
				for _ in range(num_encoder_layers)
			]
		)

		# Apply initialization of model params after building the model
		if self.apply_graphormer_init:
			self.apply(init_graphormer_params)

		def freeze_module_params(m):
			if m is not None:
				for p in m.parameters():
					p.requires_grad = False

		if freeze_embeddings:
			raise NotImplementedError("Freezing embeddings is not implemented yet.")

		for layer in range(n_trans_layers_to_freeze):
			freeze_module_params(self.layers[layer])

	def build_graphormer_graph_encoder_layer(
			self,
			embedding_dim,
			ffn_embedding_dim,
			num_attention_heads,
			dropout,
			attention_dropout,
			activation_dropout,
			activation_fn,
			export,
			q_noise,
			qn_block_size,
			pre_layernorm,
	):
		return GraphormerGraphEncoderLayer(
			embedding_dim=embedding_dim,
			ffn_embedding_dim=ffn_embedding_dim,
			num_attention_heads=num_attention_heads,
			dropout=dropout,
			attention_dropout=attention_dropout,
			activation_dropout=activation_dropout,
			activation_fn=activation_fn,
			export=export,
			q_noise=q_noise,
			qn_block_size=qn_block_size,
			pre_layernorm=pre_layernorm,
		)

	def forward(
			self,
			batched_data,
			perturb=None,
			last_state_only: bool = False,
			token_embeddings: Optional[torch.Tensor] = None,
			attn_mask: Optional[torch.Tensor] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		is_tpu = False
		# compute padding mask. This is needed for multi-head attention
		data_x = batched_data["x"]
		n_graph, n_node = data_x.size()[:2]
		# padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1
		padding_mask = (data_x.sum(-1, keepdim=True)[:, :, 0]).eq(0)
		padding_mask_cls = torch.zeros(
			n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
		)
		padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
		# B x (T+1) x 1

		if token_embeddings is not None:
			x = token_embeddings
		else:
			x = self.graph_node_feature(batched_data)

		if perturb is not None:
			# ic(torch.mean(torch.abs(x[:, 1, :])))
			# ic(torch.mean(torch.abs(perturb)))
			x[:, 1:, :] += perturb

		# x: B x T x C

		attn_bias = self.graph_attn_bias(batched_data)

		if self.embed_scale is not None:
			x = x * self.embed_scale

		if self.quant_noise is not None:
			x = self.quant_noise(x)

		if self.emb_layer_norm is not None:
			x = self.emb_layer_norm(x)

		x = self.dropout_module(x)

		# account for padding while computing the representation

		# B x T x C -> T x B x C
		x = x.transpose(0, 1)

		inner_states = []
		if not last_state_only:
			inner_states.append(x)

		for layer in self.layers:
			x, _ = layer(
				x,
				self_attn_padding_mask=padding_mask,
				self_attn_mask=attn_mask,
				self_attn_bias=attn_bias,
			)
			if not last_state_only:
				inner_states.append(x)

		graph_rep = x[0, :, :]

		if last_state_only:
			inner_states = [x]

		if self.traceable:
			return torch.stack(inner_states), graph_rep
		else:
			return inner_states, graph_rep


class GraphormerEncoder(nn.Module):
	# def __init__(self, args=None):
	def __init__(self, num_encoder_layers, multi_hop_max_dist=8, embedding_dim=80
				 , ffn_embedding=80, num_attention_heads=8, dropout=0.0, bUseDegree=False):
		super().__init__()
		# self.max_nodes = 128#args.max_nodes

		self.graph_encoder = GraphormerGraphEncoder(
			# < for graphormer
			num_atoms=4608,  # args.num_atoms,
			num_in_degree=512,  # args.num_in_degree,
			num_out_degree=512,  # args.num_out_degree,
			num_edges=1536,  # args.num_edges,
			num_spatial=512,  # args.num_spatial,
			num_edge_dis=128,  # args.num_edge_dis,
			edge_type='multi_hop',  # args.edge_type,
			multi_hop_max_dist=multi_hop_max_dist,  # args.multi_hop_max_dist,
			# >
			num_encoder_layers=num_encoder_layers,  # args.encoder_layers,
			embedding_dim=embedding_dim,  # args.encoder_embed_dim,
			ffn_embedding_dim=ffn_embedding,  # args.encoder_ffn_embed_dim,
			num_attention_heads=num_attention_heads,  # args.encoder_attention_heads,
			dropout=dropout,  # args.dropout,
			attention_dropout=0.1,  # args.attention_dropout,
			activation_dropout=0.1,  # args.act_dropout,
			encoder_normalize_before=True,  # args.encoder_normalize_before,
			pre_layernorm=False,  # args.pre_layernorm,
			apply_graphormer_init=True,  # args.apply_graphormer_init,
			activation_fn='gelu',  # args.activation_fn,
			bUseDegree=bUseDegree,
		)

		self.share_input_output_embed = False  # args.share_encoder_input_output_embed
		self.embed_out = None
		self.lm_output_learned_bias = None

		# Remove head is set to true during fine-tuning
		self.load_softmax = True  # not getattr(args, "remove_head", False)

		# self.masked_lm_pooler = nn.Linear(
		# 	args.encoder_embed_dim, args.encoder_embed_dim
		# )
		self.masked_lm_pooler = nn.Linear(embedding_dim, embedding_dim)

		# self.lm_head_transform_weight = nn.Linear(
		# 	args.encoder_embed_dim, args.encoder_embed_dim
		# )
		self.lm_head_transform_weight = nn.Linear(embedding_dim, embedding_dim)
		# self.activation_fn = utils.get_activation_fn(args.activation_fn)
		self.activation_fn = utils.get_activation_fn('gelu')
		# self.layer_norm = LayerNorm(args.encoder_embed_dim)
		self.layer_norm = LayerNorm(embedding_dim)

		self.lm_output_learned_bias = None
		if self.load_softmax:
			self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

			if not self.share_input_output_embed:
				# self.embed_out = nn.Linear(
				# 	args.encoder_embed_dim, args.num_classes, bias=False
				# )
				self.embed_out = nn.Linear(embedding_dim, 1, bias=False)
			else:
				raise NotImplementedError

	def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
		for k in batched_data.keys():
			if batched_data[k].device.type == 'cpu':
				batched_data[k] = batched_data[k].cuda()
		inner_states, graph_rep = self.graph_encoder(
			batched_data,
			perturb=perturb,
		)

		x = inner_states[-1].transpose(0, 1)

		# project masked tokens only
		if masked_tokens is not None:
			raise NotImplementedError

		return x[:,0,:]


#multi-shell Graphormer
class MultiShellGraphormer(nn.Module):
	def __init__(self, SHELLs:int, num_MLP:int=2, num_encoder_layers:int = 16, multi_hop_max_dist=8, embedding_dim=80
				 , ffn_embedding=80, num_attention_heads=8, dropout=0.0, bUseDegree=False):
		super().__init__()
		self.GraphormerEncoderList = nn.ModuleList()
		self.SHELLs = SHELLs
		self.num_encoder_layers = num_encoder_layers
		self.multi_hop_max_dist = multi_hop_max_dist
		self.embedding_dim = embedding_dim
		self.ffn_embedding = ffn_embedding
		self.num_attention_heads = num_attention_heads
		self.dropout = dropout
		self.bUseDegree = bUseDegree

		# self.MLP1 = nn.Linear(in_features=SHELLs*embedding_dim, out_features=SHELLs*embedding_dim//2)
		self.MLP = nn.Sequential()
		in_feature = SHELLs*embedding_dim
		for _ in range(num_MLP):
			out_feature = in_feature//2
			if out_feature <= 1:
				break
			self.MLP.add_module('Linear'+str(_+1), nn.Linear(in_features=in_feature, out_features=in_feature//2))
			self.MLP.add_module('ReLU'+str(_+1), nn.ReLU())
			in_feature = in_feature//2
		self.MLP.add_module('Predict', nn.Linear(in_features=in_feature, out_features=1, bias=False))

		for _ in range(SHELLs):
			Net = GraphormerEncoder(num_encoder_layers=num_encoder_layers
									, multi_hop_max_dist=multi_hop_max_dist
									, embedding_dim=embedding_dim, ffn_embedding=ffn_embedding
									, num_attention_heads=num_attention_heads, dropout=dropout, bUseDegree=bUseDegree)
			self.GraphormerEncoderList.append(Net)

	def forward(self, batched_data):
		vectorList = []
		for shell, net in enumerate(self.GraphormerEncoderList):
			shell_data = batched_data[shell+1]
			if torch.sum(shell_data['hasGraph']) != 0:
				AvailableVector = net(shell_data)
			# else:
			# 	AvailableVector = \
			# 		torch.zeros(size=(shell_data['hasGraph'].shape[0], self.ffn_embedding), requires_grad=True).cuda()
			NewVector = []
			count = 0
			for id, check in enumerate(shell_data['hasGraph']):
				if check == 1:
					NewVector.append(AvailableVector[count:count+1])
					count += 1
				else:
					NewVector.append(torch.zeros(size=(1, self.ffn_embedding), requires_grad=True).cuda())
			NewVector2 = torch.cat(NewVector, dim=0)
			vectorList.append(NewVector2)
		x = torch.cat(vectorList, dim=1)
		y = self.MLP(x)
		return y
