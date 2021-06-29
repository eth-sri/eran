"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


import numpy as np
import onnx
from onnx import numpy_helper
from config import config
import warnings

def onnxshape_to_intlist(onnxshape):
	"""
	ONNX has its own wrapper for shapes. Our optimizer expects a list of ints.

	Arguments
	---------
	onnxshape : TensorShapeProto

	Return
	------
	output : list
	    list of ints corresponding to onnxshape
	"""
	result = list(map(lambda j: 1 if j.dim_value is None else int(j.dim_value), onnxshape.dim))

	# No shape means a single value
	if not result:
		return [1]

	# convert NCHW to NHWC
	if len(result) == 4:
		return [result[0], result[2], result[3], result[1]]

	return result


def nchw_to_nhwc_shape(shape):
	"""
	Reorders dimensions of a 1D array from NCHW to NHWC, since ONNX uses NCHW, ELINA expects NHWC.

	:param index: the array to be converted

	:return: converted array
	"""
	assert len(shape) == 4, "Unexpected shape size"
	return [shape[0], shape[2], shape[3], shape[1]]


def nchw_to_nhwc_index(index: int) -> int:
	"""
	Converts an single index from NCHW to NHWC, since ONNX uses NCHW, ELINA expects NHWC,

	:param index: the index to be converted

	:return: converted index
	"""
	assert 0 <= index <= 3, f"index out of range: {index}"
	if index == 0:  # batch (N)
		return 0
	elif index == 1:  # channel (C)
		return 3
	else:
		return index - 1


def nchw_to_nhwc(array):
	"""
	ONNX uses NCHW. ELINA expects NHWC

	:param array: array to be converted

	:return: converted array
	"""
	if array.ndim == 4:
		return array.transpose(0, 2, 3, 1)

	return array




def reshape_nhwc(shape_in, shape_out):
	#print(shape_in, shape_out)
	ndim_in = len(shape_in)
	ndim_out = len(shape_out)
	total_in = np.prod(shape_in[1:ndim_in])
	total_out = np.prod(shape_out[1:ndim_out])
	assert total_in == total_out, "Reshape doesn't have same number of neurons before and after"
	array = np.asarray(range(total_in)).reshape(shape_in[1:ndim_in])
	if array.ndim == 3:
		array = array.transpose((2, 0, 1))
	array = array.reshape(shape_out[1:ndim_out])
	if array.ndim == 3:
		return array.transpose((1, 2, 0))
	else:
		return array

def prepare_model(model):
	"""
	The constructor has produced a graph_def with the help of the functions graph_util.convert_variables_to_constants and graph_util.remove_training_nodes.
	translate() takes that graph_def, imports it, and translates it into two lists which then can be processed by an Optimzer object.

	Return
	------
	(operation_types, operation_resources) : (list, list)
	    A tuple with two lists, the first one has items of type str and the second one of type dict. In the first list the operation types are stored (like "Add", "MatMul", etc.).
	    In the second list we store the resources (matrices, biases, etc.) for those operations. It is organised as follows: operation_resources[i][domain] has the resources related to
	    operation_types[i] when analyzed with domain (domain is currently either 'deepzono' or 'deeppoly', as of 8/30/18)
	"""
	shape_map = {}
	constants_map = {}
	output_node_map = {}
	input_node_map = {}

	for initial in model.graph.initializer:
		const = nchw_to_nhwc(numpy_helper.to_array(initial)).copy()
		constants_map[initial.name] = const
		shape_map[initial.name] = const.shape

	placeholdernames = []
	#print("graph ", model.graph.node)
	for node_input in model.graph.input:
		placeholdernames.append(node_input.name)
		if node_input.name not in shape_map:
			shape_map[node_input.name] = onnxshape_to_intlist(node_input.type.tensor_type.shape)
			input_node_map[node_input.name] = node_input
			
	for node in model.graph.node:
		#print(node.op_type)
		output_node_map[node.output[0]] = node
		for node_input in node.input:
			input_node_map[node_input] = node
		if node.op_type == "Flatten":
			#shape_map[node.output[0]] = shape_map[node.input[0]]
			shape_map[node.output[0]] = [1,] + [np.prod(shape_map[node.input[0]][1:]),]
		elif node.op_type == "Constant":
			const = node.attribute
			const = nchw_to_nhwc(numpy_helper.to_array(const[0].t)).copy()
			constants_map[node.output[0]] = const
			shape_map[node.output[0]] = const.shape

		elif node.op_type in ["MatMul", "Gemm"]:
			transA = 0
			transB = 0
			for attribute in node.attribute:
				if 'transA' == attribute.name:
					transA = attribute.i
				elif 'transB' == attribute.name:
					transB = attribute.i
			input_shape_A = ([1] if len(shape_map[node.input[0]])==1 else []) + list(shape_map[node.input[0]])
			input_shape_B =  list(shape_map[node.input[1]]) + ([1] if len(shape_map[node.input[1]])==1 else [])
			M = input_shape_A[transA]
			N = input_shape_B[1 - transB]
			shape_map[node.output[0]] = [M, N]

		elif node.op_type in ["Add", "Sub", "Mul", "Div"]:
			shape_map[node.output[0]] = shape_map[node.input[0]]
			if node.input[0] in constants_map and node.input[1] in constants_map:
				if node.op_type == "Add":
					result = np.add(constants_map[node.input[0]], constants_map[node.input[1]])
				elif node.op_type == "Sub":
					result = np.subtract(constants_map[node.input[0]], constants_map[node.input[1]])
				elif node.op_type == "Mul":
					result = np.multiply(constants_map[node.input[0]], constants_map[node.input[1]])
				elif node.op_type == "Div":
					result = np.divide(constants_map[node.input[0]], constants_map[node.input[1]])
				constants_map[node.output[0]] = result
		elif node.op_type in ["Conv", "MaxPool", "AveragePool"]:
			output_shape = []
			input_shape = shape_map[node.input[0]]

			require_kernel_shape = node.op_type in ["MaxPool", "AveragePool"]
			if not require_kernel_shape:
				filter_shape = shape_map[node.input[1]]
				kernel_shape = filter_shape[1:-1]

			strides = [1, 1]
			padding = [0, 0, 0, 0]
			auto_pad = 'NOTSET'
			dilations = [1, 1]
			group = 1
			ceil_mode = 0
			for attribute in node.attribute:
				if attribute.name == 'strides':
					strides = attribute.ints
				elif attribute.name == 'pads':
					padding = attribute.ints
				elif attribute.name == 'auto_pad':
					auto_pad = attribute.s
				elif attribute.name == 'kernel_shape':
					kernel_shape = attribute.ints
				elif attribute.name == 'dilations':
					dilations = attribute.ints
				elif attribute.name == 'group':
					group = attribute.i
				elif attribute.name == 'ceil_mode':
					ceil_mode = attribute.i

			effective_kernel_shape = [(kernel_shape[i] - 1) * dilations[i] + 1 for i in range(len(kernel_shape))]

			output_shape.append(input_shape[0])

			for i in range(len(kernel_shape)):
				effective_input_size = input_shape[1 + i]
				effective_input_size += padding[i]
				effective_input_size += padding[i + len(kernel_shape)]
				if ceil_mode == 1:
					strided_kernel_positions = int(np.ceil((effective_input_size - effective_kernel_shape[i]) / float(strides[i])))
				else:
					strided_kernel_positions = int(np.floor((effective_input_size - effective_kernel_shape[i]) / strides[i]))
				output_shape.append(1 + strided_kernel_positions)

			if require_kernel_shape:
				output_shape.append(input_shape[3])
			else:
				output_shape.append(filter_shape[0])

			shape_map[node.output[0]] = output_shape
		elif node.op_type in ["Relu", "Sigmoid", "Tanh", "Softmax", "BatchNormalization", "LeakyRelu"]:
			shape_map[node.output[0]] = shape_map[node.input[0]]

		# Gather is for the moment solely for shapes
		elif node.op_type == "Gather":
			axis = 0
			for attribute in node.attribute:
				axis = attribute.i
			if node.input[0] in constants_map and node.input[1] in constants_map:
				data = constants_map[node.input[0]]
				indexes = constants_map[node.input[1]]
				constants_map[node.output[0]] = np.take(data, indexes, axis)

			if node.input[0] in shape_map and node.input[1] in shape_map:
				r = len(shape_map[node.input[0]])
				q = len(shape_map[node.input[1]])
				out_rank = q + r - 1
				if out_rank == 0:
					shape_map[node.output[0]] = shape_map[node.input[1]]
				else:
					output_shape = []
					for i in range(out_rank):
						if i < axis:
							output_shape.append(shape_map[node.input[0]][i]) # i < axis < r
						elif i >= axis and i < axis + q:
							output_shape.append(shape_map[node.input[0]][i-axis]) # i - axis < q
						else:
							output_shape.append(shape_map[node.input[0]][i - q + 1]) # i < out_rank < q + r - 1
					shape_map[node.output[0]] = output_shape
		elif node.op_type == "Shape":
			if node.input[0] in shape_map:
				constants_map[node.output[0]] = shape_map[node.input[0]]
				shape_map[node.output[0]] = [len(shape_map[node.input[0]])]

		#elif node.op_type == "Cast":
			#shape_map[node.output[0]] = shape_map[node.input[0]]
			#print("CASTING ", node.input[0], shape_map[node.input[0]], shape_map[node.output[0]])

		elif node.op_type == "Reshape":
			#print("RESHAPE ", node.input, node.output)
			if node.input[1] in constants_map:
				total = 1
				replace_index = -1
				for index in range(len(constants_map[node.input[1]])):
					if constants_map[node.input[1]][index] == -1:
						replace_index = index
					else:
						total *= constants_map[node.input[1]][index]

				if replace_index != -1:
					constants_map[node.input[1]][replace_index] = np.prod(shape_map[node.input[0]]) / total

				if len(constants_map[node.input[1]]) == 4:
					shape_map[node.output[0]] = [constants_map[node.input[1]][0], constants_map[node.input[1]][2], constants_map[node.input[1]][3], constants_map[node.input[1]][1]]
				else:
					shape_map[node.output[0]] = constants_map[node.input[1]]

		elif node.op_type == "Unsqueeze":
			if node.input[0] in shape_map:
				axis = node.attribute[0].ints
				output_shape = list(shape_map[node.input[0]])
				if node.input[0] in constants_map:
					constants_map[node.output[0]] = constants_map[node.input[0]]
				for i in axis:
					output_shape.insert(i, 1)
					if node.input[0] in constants_map:
						constants_map[node.output[0]] = np.expand_dims(constants_map[node.output[0]], axis=i)
				shape_map[node.output[0]] = output_shape

		elif node.op_type == "Concat":
			all_constant = True
			n_dim = len(shape_map[node.input[0]])
			if n_dim > 2:
				axis = nchw_to_nhwc_index(node.attribute[0].i)
			else:
				axis = node.attribute[0].i
			for node_input in node.input:
				if not node_input in constants_map:
					all_constant = False
					break
			if all_constant:
				constants_map[node.output[0]] = np.concatenate([constants_map[input] for input in node.input], axis=axis)
			all_shape_known = True
			for node_input in node.input:
				if not node_input in shape_map:
					all_shape_known = False
					break
			assert all_shape_known, "Unknown shape for at least one node input!"
			new_axis_size = 0
			for node_input in node.input:
				new_axis_size += shape_map[node_input][axis]
			shape_map[node.output[0]] = [shape_map[node.input[0]][i] if i != axis else new_axis_size for i in range(len(shape_map[node.input[0]]))]
			if not all_constant:
				assert axis == n_dim-1, "ELINA currently only supports concatenation on the channel dimension"

		elif node.op_type == "Tile":
			repeats = nchw_to_nhwc_shape(constants_map[node.input[1]])
			input_shape = list(shape_map[node.input[0]])
			assert len(repeats) == len(input_shape), "Expecting one repeat factor per dimension"
			output_shape = [factor * size for factor, size in zip(repeats, input_shape)]
			shape_map[node.output[0]] = output_shape

			repeat_index = np.where(np.array(repeats) != 1)[0]
			assert len(repeat_index) == 1, "ELINA backend currently only supports repeats for one dimension"
			repeat_index = repeat_index.item()
			assert repeat_index == 1, "ELINA backend currently only supports repeats for the first dimension"
			assert input_shape[0] == 1, "ELINA backend currently only supports repeats for dimensions of size 1"

		elif node.op_type == "Expand":
			if node.input[1] in constants_map:
				if len(constants_map[node.input[1]]) == 4:
					shape_map[node.output[0]] = [constants_map[node.input[1]][0], constants_map[node.input[1]][2], constants_map[node.input[1]][3], constants_map[node.input[1]][1]]
				else:
					shape_map[node.output[0]] = constants_map[node.input[1]]

				result = np.zeros(shape_map[node.output[0]]) + constants_map[node.input[0]]
				constants_map[node.output[0]] = result
		elif node.op_type == "Pad":
			input_shape = np.array(shape_map[node.input[0]])
			for attribute in node.attribute:
				if attribute.name == "pads":
					padding = np.array(attribute.ints)
				if attribute.name == "mode":
					assert attribute.s == bytes(b'constant'), "only zero padding supported"
				if attribute.name == "value":
					assert attribute.f == 0, "only zero padding supported"
			output_shape = np.copy(input_shape)
			input_dim = len(input_shape)
			assert len(padding) == 2* input_dim
			for i in range(2,input_dim): # only pad spatial dimensions
				output_shape[i-1] += padding[i]+padding[i+input_dim]
			shape_map[node.output[0]] = list(output_shape)
		else:
			assert 0, f"Operations of type {node.op_type} are not yet supported."

	#print('const_map')
	#print(constants_map)
	#print('shape_map')
	#print(shape_map)
	return shape_map, constants_map, output_node_map, input_node_map, placeholdernames


class ONNXTranslator:
	"""
	This class is used to turn a ONNX model into two lists that then can be processed by an Optimizer object
	"""	
	def __init__(self, model, is_gpupoly):
		"""
		This constructor takes a reference to a ONNX Model and checks model, infers intermediate shapes and sets up maps from name to type and node or constant value
		graph_util.convert_variables_to_constants and graph_util.remove_training_nodes to cleanse the graph of any nodes that are linked to training. This leaves us with 
		the nodes you need for inference. 
		In the resulting graph there should only be tf.Operations left that have one of the following types [Const, MatMul, Add, BiasAdd, Conv2D, Reshape, MaxPool, AveragePool, Placeholder, Relu, Sigmoid, Tanh, LeakyRelu]
		If the input should be a Keras model we will ignore operations with type Pack, Shape, StridedSlice, and Prod such that the Flatten layer can be used.
		
		Arguments
		---------
		model : onnx.ModelProto
		"""
		if issubclass(model.__class__, onnx.ModelProto):
			onnx.checker.check_model(model)
			self.model = model
			self.nodes = self.model.graph.node
			self.is_gpupoly = is_gpupoly
			self.shape_map, self.constants_map, self.output_node_map, self.input_node_map, self.placeholdernames = prepare_model(model)
		else:
			assert 0, 'not onnx model'

	def find_input(self):
		inputs_dir = {x.name: x for x in self.model.graph.input}
		all_inputs = [x for y in self.nodes for x in y.input]
		[all_inputs.remove(x) for y in self.nodes for x in y.output if x in all_inputs]
		[all_inputs.remove(x.name) for x in self.model.graph.initializer if x.name in all_inputs]

		assert all_inputs[0] in inputs_dir

		return inputs_dir[all_inputs[0]]

	@staticmethod
	def clean_shape(shape_raw):
		'''
		Onnx translator expects the inputs and outputs of each node to not have 0-sized dimensions.
		These can occur, if other formats are converted to onnx instead of directly exporting an onnx model.
		This function handles such occurances, setting the 0-sized dimension to 1.

		Arguments
		--------
		shape_raw : A shape in form of a list
		'''
		shape_cleaned = [1 if x == 0 else x for x in shape_raw]
		if 0 in shape_raw:
			warnings.warn(f"0-sized dimension encountered: {shape_raw} and changed to: {shape_cleaned}",RuntimeWarning)
		return shape_cleaned
		
	def translate(self):
		"""
		The constructor has produced a graph_def with the help of the functions graph_util.convert_variables_to_constants and graph_util.remove_training_nodes.
		translate() takes that graph_def, imports it, and translates it into two lists which then can be processed by an Optimzer object.
		
		Return
		------
		(operation_types, operation_resources) : (list, list)
		    A tuple with two lists, the first one has items of type str and the second one of type dict. In the first list the operation types are stored (like "Add", "MatMul", etc.).
		    In the second list we store the resources (matrices, biases, etc.) for those operations. It is organised as follows: operation_resources[i][domain] has the resources related to
		    operation_types[i] when analyzed with domain (domain is currently either 'deepzono' or 'deeppoly', as of 8/30/18)
		"""
		operation_types     = ["Placeholder"]
		# placeholder = self.model.graph.input[0]
		placeholder = self.find_input()
		in_out_placeholder = ([], placeholder.name, self.clean_shape(onnxshape_to_intlist(placeholder.type.tensor_type.shape)))
		operation_resources = [{'deepzono':in_out_placeholder, 'deeppoly':in_out_placeholder}]
		reshape_map = {}
		operations_to_be_ignored = ["Pack", "Shape", "StridedSlice", "Prod", "Unsqueeze", "Softmax", "Concat", "Flatten", "BatchNormalization"]
		padding_merger_dict = {}


		### Check if there Are Add/Sub and Div/Mul layers that can be interpreted as normalization layer
		stop_norm_layers = ["MatMul","Gemm","Conv","MaxPool","Relu","Sigmoid","Tanh","LeakyRelu"]
		stop_norm_layer = len(self.nodes)
		extract_mean = False
		extract_std = False
		for node_idx, node in enumerate(self.nodes):
			if node.op_type in stop_norm_layers or (extract_mean and extract_std):
				stop_norm_layer = node_idx
				break
			if node.op_type in ["Add","Sub"]:
				extract_mean = True
			elif node.op_type in ["Div", "Mul"]:
				extract_std = True
		extract_norm = extract_std and extract_mean

		for node_idx, node in enumerate(self.nodes):
			# print("node ", node.op_type)
			if node.op_type == "Constant":
				continue
			elif node.op_type in operations_to_be_ignored:
				input_name  = node.input[0]
				output_name = node.output[0]
				if input_name in reshape_map:
					reshape_map[output_name] = reshape_map[input_name]
				else:
					reshape_map[output_name] = input_name
				continue

			operation_types.append(node.op_type)
			# take means and stds out of the network
			if extract_norm and node_idx <= stop_norm_layer and len(operation_types) == 2 and node.op_type in ["Add", "Sub", "Mul", "Div"] and node.output[0] not in self.constants_map:
				constant = self.add_resources(node)[0].reshape(-1)
				if node.op_type == "Add":
					config.mean = np.multiply(constant, -1)
					print(f"Mean of {config.mean} extracted from network")
				elif node.op_type == "Sub":
					config.mean = constant
					print(f"Mean of {config.mean} extracted from network")
				elif node.op_type == "Mul":
					config.std = np.divide(1, constant)
					print(f"Std of {config.std} extracted from network")
				elif node.op_type == "Div":
					config.std = constant
					print(f"Std of {config.std} extracted from network")

				self.ignore_node(node, operation_types, reshape_map)
				continue

			input_onnx_names = []
			for name in node.input:
				kind = self.get_kind(name)
				if name in reshape_map:
					name = reshape_map[name]
				if kind == 'Constant':
					continue
				input_onnx_names.append(name)
			shape = self.get_shape(node.output[0])
			shape = self.clean_shape(shape)
			in_out_info = (input_onnx_names, node.output[0], shape)

			if node.op_type == "MatMul":
				deeppoly_res = self.matmul_resources(node) + in_out_info
				deepzono_res = deeppoly_res
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type == "Gemm":
				deeppoly_res = self.gemm_resources(node) + in_out_info
				deepzono_res = deeppoly_res
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type in ["Add", "Mul"]:
				left_type  = self.get_kind(node.input[0])
				right_type = self.get_kind(node.input[1])
				if left_type == 'Constant' and right_type == 'Constant':
					operation_types.pop()
				elif left_type == 'Constant' or right_type == 'Constant':
					deeppoly_res = self.add_resources(node) + in_out_info
					deepzono_res = deeppoly_res
					operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
				else:
					if node.op_type != "Add":
						assert 0, "we don't support residual operations other then add"
					operation_types[-1] = "Resadd"
					operation_resources.append({'deepzono':in_out_info, 'deeppoly':in_out_info})

			elif node.op_type == "Sub":
				left_type  = self.get_kind(node.input[0])
				right_type = self.get_kind(node.input[1])
				if left_type == 'Constant' and right_type == 'Constant':
					assert 0, "we don't support the subraction of two constants yet"
				elif left_type == 'Constant' or right_type == 'Constant':
					deeppoly_res = self.sub_resources(node) + in_out_info
					deepzono_res = deeppoly_res
					operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
				else:
					assert 0, "we don't support the ressub yet"
					operation_types[-1] = "Ressub"
					operation_resources.append({'deepzono':in_out_info, 'deeppoly':in_out_info})
			elif node.op_type == "Conv":
				filters, bias, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, kernel_shape = self.conv_resources(node)
				if node.name in padding_merger_dict:
					image_shape, in_out_info, pad_top, pad_left, pad_bottom, pad_right = self.merge_padding(node, padding_merger_dict, in_out_info, pad_top, pad_left, pad_bottom, pad_right)
				deeppoly_res = (filters, bias, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right) + in_out_info
				deepzono_res = deeppoly_res
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type == "Pad":
				image_shape, pad_top, pad_left, pad_bottom, pad_right = self.pad_resources(node)
				deeppoly_res = (image_shape, pad_top, pad_left, pad_bottom, pad_right) + in_out_info
				deepzono_res = deeppoly_res
				consequent_nodes = [node_i for node_i in self.nodes if node.output[0] in node_i.input]
				can_be_merged = all([node_i.op_type in ["Conv"] for node_i in consequent_nodes])
				if can_be_merged:
					padding_merger_dict.update({node_i.name: deeppoly_res for node_i in consequent_nodes})
					self.ignore_node(node, operation_types, reshape_map)
				else:
					operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type == "MaxPool" or node.op_type == "AveragePool":
				image_shape, kernel_shape, strides, padding, dilations, pad_top, pad_left, pad_bottom, pad_right, ceil_mode, storage_order = self.pool_resources(node)
				if node.name in padding_merger_dict:
					image_shape, in_out_info, pad_top, pad_left, pad_bottom, pad_right = self.merge_padding(node, padding_merger_dict, in_out_info, pad_top, pad_left, pad_bottom, pad_right)
				deeppoly_res = (image_shape, kernel_shape, strides, pad_top, pad_left, pad_bottom, pad_right) + in_out_info
				# TODO padding is expected to be string in tf. dilations, auto_pad, ceil_mode, storage_order are unused at the moment
				deepzono_res = deeppoly_res
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type == "Placeholder":
				assert 0, "Placeholder is not in the ONNX graph"
			elif node.op_type in ["Relu", "Sigmoid", "Tanh", "LeakyRelu"]:
				deeppoly_res = self.nonlinearity_resources(node) + in_out_info
				deepzono_res = deeppoly_res
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})

			# Gather is for the moment solely for shapes
			elif node.op_type == "Gather":
				only_shape, image_shape, indexes, axis = self.gather_resources(node)
				
				if only_shape:
					self.ignore_node(node, operation_types, reshape_map)
				else:
					deeppoly_res = (image_shape, indexes, axis) + in_out_info
					deepzono_res = deeppoly_res
					operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})

			elif node.op_type == "Expand":
				only_shape, image_shape, to_expand = self.expand_resources(node)
				if only_shape:
					operation_types.pop()
				else:
					deeppoly_res = (image_shape, indexes, axis) + in_out_info
					deepzono_res = deeppoly_res
					operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})

			elif node.op_type == "Reshape":
				if node.output[0] in self.input_node_map and self.input_node_map[node.output[0]].op_type in ["MatMul", "Gemm"]:
					
					self.ignore_node(node, operation_types, reshape_map)

				elif node.output[0] in self.input_node_map and self.input_node_map[node.output[0]].op_type in ["Relu", "Sigmoid", "Tanh", "LeakyRelu"] and self.input_node_map[self.input_node_map[node.output[0]].output[0]].op_type == "Reshape":
					# ignore this reshape even in the shape_map
					self.shape_map[node.output[0]] = self.shape_map[node.input[0]]
					self.shape_map[self.input_node_map[node.output[0]].output[0]] = self.shape_map[node.input[0]]
					self.ignore_node(node, operation_types, reshape_map)
				else:
					shape_in = self.get_shape(node.input[0])
					shape_out = self.get_shape(node.output[0])
					if len(shape_in) == 2 and len(shape_out) == 2:
						self.ignore_node(node, operation_types, reshape_map)
					else:
						indexes = reshape_nhwc(shape_in, shape_out)
						deeppoly_res = (indexes,) + in_out_info
						deepzono_res = deeppoly_res
						operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})

			elif node.op_type == "Concat":
				n_dim = len(self.shape_map[node.input[0]])
				if n_dim > 2:
					axis = nchw_to_nhwc_index(node.attribute[0].i)
				else:
					axis = node.attribute[0].i
				assert axis == n_dim - 1, "ELINA backend currently only supports concatenation on the channel dimension"
				channels = []
				for input_node in node.input:
					channels.append(self.get_shape(input_node)[axis])
				# width = shape[1]
				# height = shape[2]
				operation_resources.append({'deeppoly': (channels,) + in_out_info})

			elif node.op_type == "Tile":
				repeats = nchw_to_nhwc_shape(self.constants_map[node.input[1]])
				repeat_factor = repeats[repeats != 1].item()
				operation_resources.append({'deeppoly': (repeat_factor,) + in_out_info})

			else:
				assert 0, "Operations of type " + node.op_type + " are not yet supported."

			assert all([0 not in y[-1] for x in operation_resources for y in x.values()]), "Ensure inputs and outpus include no dimensions of size 0"

		return operation_types, operation_resources

	def ignore_node(self, node, operation_types, reshape_map):
		operation_types.pop()
		input_name = node.input[0]
		#print("ignore ", len(node.input), reshape_map)
		output_name = node.output[0]
		if input_name in reshape_map:
			reshape_map[output_name] = reshape_map[input_name]
		else:
			reshape_map[output_name] = input_name

	def merge_padding(self, node, padding_merger_dict, in_out_info, pad_top, pad_left, pad_bottom, pad_right):
		image_shape, m_pad_top, m_pad_left, m_pad_bottom, m_pad_right, input_node, _, _ = padding_merger_dict[node.name]
		in_out_info = (input_node, in_out_info[1], in_out_info[2])
		pad_top += m_pad_top
		pad_left += m_pad_left
		pad_bottom += m_pad_bottom
		pad_right += m_pad_right
		return image_shape, in_out_info, pad_top, pad_left, pad_bottom, pad_right


	def get_kind(self, name):
		if name in self.constants_map:
			kind = 'Constant'
		elif name in self.placeholdernames:
			kind = 'Placeholder'
		else:
			kind = self.output_node_map[name].op_type
		return kind


	def get_shape(self, name):
		if name in self.shape_map:
			return self.shape_map[name]


	def matmul_resources(self, node):
		"""
		checks which one of the direct ancestor tf.Operations is a constant and returns the underlying onnx as a numpy.ndarray inside a tuple. The matrix is manipulated in a way that it can be
		used as the left multiplier in the matrix multiplication.

		Arguments
		---------
		node : ONNX.Node
		    must have op_type "MatMul"

		Return
		------
		output : tuple
		    tuple with the matrix (of type numpy.ndarray) as its only item
		"""
		inputs = node.input
		left   = inputs[0]
		right  = inputs[1]


		if left in self.constants_map:
			matrix = self.constants_map[left]
			matrix = self.reshape_adjust(right, matrix, True)
		else:
			matrix = self.constants_map[right].transpose()
			matrix = self.reshape_adjust(left, matrix)
		return matrix,

	def reshape_adjust(self, element, matrix, is_right=False):
		if self.get_kind(element) in ['Reshape', 'Flatten'] and not self.is_gpupoly: #TODO check whether it should be triggered for Flatten layers to
			shape_in = self.get_shape(self.output_node_map[element].input[0])
			shape_out = self.get_shape(self.output_node_map[element].output[0])
			if config.debug:
				print('reshape adjust ', str(shape_in), 'to', str(shape_out))
			
			indexes = reshape_nhwc(shape_in, shape_out)
			#indexes = indexes[0]
			inverse_perm = np.arange(len(indexes))[np.argsort(indexes)]
			if is_right:
				matrix = matrix[inverse_perm, :]
			else:
				matrix = matrix[:, inverse_perm]
		return matrix

	def gemm_resources(self, node):
		"""
		checks which one of the direct ancestor tf.Operations is a constant and returns the underlying onnx as a numpy.ndarray inside a tuple. The matrix is manipulated in a way that it can be
		used as the left multiplier in the matrix multiplication.

		Arguments
		---------
		node : ONNX.Node
		    must have op_type "Gemm"

		Return
		------
		output : tuple
		    tuple with the matrix and bias (of type numpy.ndarray) and is_left used to calculate the output shape
		"""
		inputs = node.input
		left   = inputs[0]
		right  = inputs[1]
		bias  = self.constants_map[inputs[2]]

		transA = False
		transB = False
		alpha = 1.0
		beta = 1.0
		for att in node.attribute:
			if 'transA' == att.name:
				transA = att.i == 1
			elif 'transB' == att.name:
				transB = att.i == 1
			elif 'alpha' == att.name:
				alpha = att.f
			elif 'beta' == att.name:
				beta = att.f
			else:
				assert 0, "Unkown attribute " + att.name + " for operation type " + node.op_type


		if left in self.constants_map:
			matrix = self.constants_map[left] if not transA else self.constants_map[left].transpose()
			matrix = self.reshape_adjust(right, matrix, True)
		else:
			matrix = self.constants_map[right].transpose() if not transB else self.constants_map[right]
			matrix = self.reshape_adjust(left, matrix)
		return matrix * alpha, bias * beta

	def add_resources(self, node):
		"""
		checks which one of the direct ancestor tf.Operations is a constant and returns the underlying onnx as a numpy.ndarray inside a tuple.

		Arguments
		---------
		node : ONNX.Node
		    must have op_type "Add"

		Return
		------
		output : tuple
		    tuple with the addend (of type numpy.ndarray) as its only item
		"""
		inputs = node.input
		left = inputs[0]
		right = inputs[1]

		if left in self.constants_map:
			addend = self.constants_map[left]
		else:
			addend = self.constants_map[right]
		return addend,


	def sub_resources(self, node):
		"""
		checks which one of the direct ancestors is a constant and returns the underlying onnx as a numpy.ndarray and a bool is_minuend, whether the returned ndarray is the minuend, inside a tuple.

		Arguments
		---------
		node : ONNX.Node
		    must have op_type "Sub"

		Return
		------
		output : tuple
		    tuple with the addend (of type numpy.ndarray) and left_constant
		"""
		inputs = node.input
		left   = inputs[0]
		right  = inputs[1]

		if left in self.constants_map:
			addend = self.constants_map[left]
			is_minuend = True
		else:
			addend = self.constants_map[right]
			is_minuend = False
		return addend, is_minuend
		
		
	def conv_resources(self, node):
		"""
		Extracts the filter, the stride of the filter, and the padding from node as well as the shape of the input coming into node
		
		Arguments
		---------
		node : ONNX.Node
		    must have op_type "Conv"
		
		Return 
		------
		output : tuple
		    has 4 entries (numpy.ndarray, numpy.ndarray, numpy.ndarray, str)
		"""
		inputs  = node.input
		image   = inputs[0]
		filters = self.constants_map[node.input[1]].transpose(1, 2, 3, 0)
		if len(node.input) == 3:
			bias = self.constants_map[node.input[2]]
		else:
			bias = np.zeros(filters.shape[3])
		image_shape = self.get_shape(image)[1:]
		pads = [0, 0, 0, 0]
		for attribute in node.attribute:
			if attribute.name == 'strides':
				strides = attribute.ints
			elif attribute.name == 'pads':
				pads = attribute.ints
			elif attribute.name == 'kernel_shape':
				kernel_shape = attribute.ints

		pad_top = pads[0]
		pad_left = pads[1]
		pad_bottom = pads[2]
		pad_right = pads[3]
		# assert pad_top == pad_bottom, 'different padding for top and bottom is not supported in ERAN'
		# assert pad_left == pad_right, 'different padding for left and right is not supported in ERAN'
		return filters, bias, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, kernel_shape

	def pad_resources(self, node):
		"""
		Extracts the padding from node as well as the shape of the input coming into node

		Arguments
		---------
		node : ONNX.Node
		    must have op_type "Pad"

		Return
		------
		output : tuple
		    has 4 entries (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
		"""
		inputs = node.input
		image = inputs[0]
		image_shape = self.get_shape(image)[1:]

		pads = [0, 0, 0, 0]
		for attribute in node.attribute:
			if attribute.name == 'pads':
				pads = attribute.ints

		pad_top = pads[2]
		pad_left = pads[3]
		pad_bottom = pads[6]
		pad_right = pads[7]
		return image_shape, pad_top, pad_left, pad_bottom, pad_right
	
	
	def pool_resources(self, node):
		"""
		Extracts the incoming image size (heigth, width, channels), the size of the maxpool/averagepool window (heigth, width), and the strides of the window (heigth, width)
		
		Arguments
		---------
		node : ONNX.Node
		    must have op_type "MaxPool" or "AveragePool"
		
		Return
		------
		output : tuple
		    has 4 entries - (list, numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int, str)
		"""
		image       = node.input[0]
		
		image_shape = self.get_shape(image)[1:]

		padding = 'NOTSET'
		ceil_mode = 0
		storage_order = 0
		pads = [0, 0, 0, 0]
		dilations = None

		for attribute in node.attribute:
			if attribute.name == 'kernel_shape':
				kernel_shape = attribute.ints
			if attribute.name == 'strides':
				strides = attribute.ints
			elif attribute.name == 'pads':
				pads = attribute.ints
			elif attribute.name == 'dilations':
				dilations = attribute.ints
			elif attribute.name == 'auto_pad':
				padding = attribute.s
			elif attribute.name == 'ceil_mode':
				ceil_mode = attribute.i
			elif attribute.name == 'storage_order':
				storage_order = attribute.i
		pad_top = pads[0]
		pad_left = pads[1]
		pad_bottom = pads[2]
		pad_right = pads[3]
		assert pad_top == pad_bottom, 'different padding for top and bottom is not supported in ERAN'
		assert pad_left == pad_right, 'different padding for left and right is not supported in ERAN'
		return image_shape, kernel_shape, strides, padding, dilations, pad_top, pad_left, pad_bottom, pad_right, ceil_mode, storage_order
	
	
	def nonlinearity_resources(self, op):
		"""
		This function only outputs an empty tuple, to make the code look more consistent
		
		Return
		------
		output : tuple
		    but is empty
		"""
		return ()


	def gather_resources(self, node):
		"""
		Extracts the indexes in the image which have to be gathered.

		Arguments
		---------
		node : ONNX.Node
		    must have op_type "Gather"

		Return
		------
		output : tuple
		    has 4 entries - (list, numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int, str)
		"""
		inputs  = node.input
		image   = inputs[0]
		if node.output[0] in self.constants_map:
			only_shape = True
			image_shape, indexes, axis = None, None, None
		else:
			only_shape = False
			image_shape = self.get_shape(image)[1:]
			indexes = self.constants_map[node.input[1]]
			axis = node.attribute[0].i
		return only_shape, image_shape, indexes, axis


	def expand_resources(self, node):
		if node.output[0] in self.constants_map:
			only_shape = True
			image_shape, to_expand = None, None
		else:
			assert 0, "Implementation for 'Expand' is missing."
		return only_shape, image_shape, to_expand
