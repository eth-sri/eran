'''
@author: Adrian Hoffmann
'''
import numpy as np
import onnx
from onnx import shape_inference
from onnx import numpy_helper


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
	if not result:
		return [1]
	return result



class ONNXTranslator:
	"""
	This class is used to turn a ONNX model into two lists that then can be processed by an Optimizer object
	"""	
	def __init__(self, model):
		"""
		This constructor takes a reference to a ONNX Model and checks model, infers intermediate shapes and sets up maps from name to type and node or constant value
		graph_util.convert_variables_to_constants and graph_util.remove_training_nodes to cleanse the graph of any nodes that are linked to training. This leaves us with 
		the nodes you need for inference. 
		In the resulting graph there should only be tf.Operations left that have one of the following types [Const, MatMul, Add, BiasAdd, Conv2D, Reshape, MaxPool, Placeholder, Relu, Sigmoid, Tanh]
		If the input should be a Keras model we will ignore operations with type Pack, Shape, StridedSlice, and Prod such that the Flatten layer can be used.
		
		Arguments
		---------
		model : onnx.ModelProto
		"""
		if issubclass(model.__class__, onnx.ModelProto):
			onnx.checker.check_model(model)
			inferred_model = shape_inference.infer_shapes(model)
			onnx.checker.check_model(inferred_model)
			self.model = inferred_model
			self.nodes = self.model.graph.node

			value_node_map = {}
			for node in self.nodes:
				for value in node.output:
					value_node_map[value] = node
			self.value_node_map = value_node_map

			constants_map = {}
			for initial in self.model.graph.initializer:
				constants_map[initial.name] = numpy_helper.to_array(initial)
			self.constants_map = constants_map

			value_info_map = {}
			for info in self.model.graph.value_info:
				value_info_map[info.name] = info.type.tensor_type
			for info in self.model.graph.input:
				value_info_map[info.name] = info.type.tensor_type
			for info in self.model.graph.output:
				value_info_map[info.name] = info.type.tensor_type
			self.value_info_map = value_info_map
		else:
			assert 0, 'not onnx model'
	
	
		
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
		placeholder = self.model.graph.input[0]
		in_out_placeholder = ([], placeholder.name, onnxshape_to_intlist(placeholder.type.tensor_type.shape))
		operation_resources = [{'deepzono':in_out_placeholder, 'deeppoly':in_out_placeholder}]
		reshape_map = {}
		operations_to_be_ignored = ["Reshape", "Pack", "Shape", "StridedSlice", "Prod", "Concat"]
		ignore_for_now = ['Gather', 'Unsqueeze']
		operations_to_be_ignored += ignore_for_now

		for node in self.nodes:
			if node.op_type == "Constant":
				continue
			elif node.op_type in operations_to_be_ignored:
				input_name  = node.input[0]
				output_name = node.output[0]
				kind = self.get_kind(input_name)
				if kind in operations_to_be_ignored:
					reshape_map[output_name] = reshape_map[input_name]
				else:
					reshape_map[output_name] = input_name
				continue

			operation_types.append(node.op_type)
			input_onnx_names = []
			for inp in node.input:
				name = inp
				kind = self.get_kind(inp)
				if kind in operations_to_be_ignored:
					name = reshape_map[name]
				if kind == 'Constant':
					continue
				input_onnx_names.append(name)
			in_out_info = (input_onnx_names, node.output[0], onnxshape_to_intlist(self.value_info_map[node.output[0]].shape))

			if node.op_type == "MatMul":
				deeppoly_res = self.matmul_resources(node) + in_out_info
				deepzono_res = deeppoly_res
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type == "Gemm":
				deeppoly_res = self.gemm_resources(node) + in_out_info
				deepzono_res = deeppoly_res
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type == "Add":
				left_type  = self.get_kind(node.input[0])
				right_type = self.get_kind(node.input[1])
				if left_type == 'Constant' and right_type == 'Constant':
					assert 0, "we don't support the addition of two constants yet"
				elif left_type == 'Constant' or right_type == 'Constant':
					deeppoly_res = self.add_resources(node) + in_out_info
					deepzono_res = deeppoly_res
					operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
				else:
					operation_types[-1] = "Resadd"
					operation_resources.append({'deepzono':in_out_info, 'deeppoly':in_out_info})
			elif node.op_type == "BiasAdd":
				if self.get_kind(node.input[1]) == 'Constant':
					deeppoly_res = self.add_resources(node) + in_out_info
					deepzono_res = deeppoly_res
					operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
				else:
					assert 0, "this bias add doesn't meet our assumption (bias is constant)"
			elif node.op_type == "Conv":
				filters, bias, image_shape, strides, padding = self.conv_resources(node)
				deeppoly_res = (filters, bias, image_shape, strides, padding) + in_out_info
				deepzono_res = deeppoly_res
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type == "MaxPool":
				image_shape, kernel_shape, strides, padding, dilations, auto_pad, ceil_mode, storage_order = self.maxpool_resources(node)
				deeppoly_res =  (image_shape, kernel_shape, in_out_info[2]) + in_out_info
				# TODO padding is expected to be string in tf. dilations, auto_pad, ceil_mode, storage_order are unused at the moment
				deepzono_res = (image_shape, kernel_shape, strides, padding) + in_out_info
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type == "Placeholder":
				assert 0, "Placeholder is not in the ONNX graph"
				deeppoly_res = in_out_info
				deepzono_res = in_out_info
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			elif node.op_type in ["Relu", "Sigmoid", "Tanh"]:
				deeppoly_res = self.nonlinearity_resources(node) + in_out_info
				deepzono_res = deeppoly_res
				operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			#elif op.type == "ConcatV2":
			#	print("Concatv2")
			#	deeppoly_res = self.concat_resources(op)
			#	deepzono_res = deeppoly_res + in_out_info
			#	operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
			else:
				print("operation type1 ",in_out_info, self.value_info_map[node.input[0]].shape, self.value_info_map[node.input[1]].shape)
				assert 0, "Operations of type " + node.op_type + " are not yet supported."

		return operation_types, operation_resources


	def get_kind(self, name):
		if 'weight' in name or 'bias' in name:
			kind = 'Constant'
		elif 'input' in name:
			kind = 'Placeholder'
		else:
			kind = self.value_node_map[name].op_type
		return kind


	def gemm_resources(self, op):
		"""
		checks which one of the direct ancestor tf.Operations is a constant and returns the underlying onnx as a numpy.ndarray inside a tuple. The matrix is manipulated in a way that it can be
		used as the left multiplier in the matrix multiplication.

		Arguments
		---------
		op : ONNX.Node
		    must have op_type "Gemm"

		Return
		------
		output : tuple
		    tuple with the matrix (of type numpy.ndarray) as its only item
		"""
		print(op)
		inputs = op.input
		left   = inputs[0]
		right  = inputs[1]
		bias  = self.constants_map[inputs[2]]
		print(op)

		transA = False
		transB = False
		alpha = 1.0
		beta = 1.0
		for att in op.attribute:
			if 'transA' == att.name:
				transA = att.i == 1
			elif 'transB' == att.name:
				transB = att.i == 1
			elif 'alpha' == att.name:
				alpha = att.f
			elif 'beta' == att.name:
				beta = att.f
			else:
				assert 0, "Unkown attribute " + att.name + " for operation type " + op.op_type


		if left in self.constants_map:
			matrix = self.constants_map[left] if not transA else self.constants_map[left].transpose()
		else:
			matrix = self.constants_map[right] if not transB else self.constants_map[right].transpose()
		return matrix * alpha, bias * beta
	
	
	def add_resources(self, op):
		"""
		checks which one of the direct ancestor tf.Operations is a constant and returns the underlying onnx as a numpy.ndarray inside a tuple.
		
		Arguments
		---------
		op : ONNX.Node
		    must have op_type "Add"
		
		Return 
		------
		output : tuple
		    tuple with the addend (of type numpy.ndarray) as its only item   
		"""
		inputs = op.inputs
		left   = inputs[0]
		right  = inputs[1]
		
		if left in self.constants_map:
			addend = self.constants_map[left]
		else:
			addend = self.constants_map[right]
		return addend,
		
		
	def conv_resources(self, op):
		"""
		Extracts the filter, the stride of the filter, and the padding from op as well as the shape of the input coming into op
		
		Arguments
		---------
		op : ONNX.Node
		    must have op_type "Conv"
		
		Return 
		------
		output : tuple
		    has 4 entries (numpy.ndarray, numpy.ndarray, numpy.ndarray, str)
		"""
		inputs  = op.input
		image   = inputs[0]
		filters = self.constants_map[op.input[1]]
		bias = self.constants_map[op.input[2]]

		image_shape = onnxshape_to_intlist(self.value_info_map[image].shape)
		for attribute in op.attribute:
			if attribute.name == 'strides':
				strides = attribute.ints
			elif attribute.name == 'pads':
				padding = attribute.ints
		return filters, bias, image_shape, strides, padding
	
	
	def maxpool_resources(self, op):
		"""
		Extracts the incoming image size (heigth, width, channels), the size of the maxpool window (heigth, width), and the strides of the window (heigth, width)
		
		Arguments
		---------
		op : ONNX.Node
		    must have op_type "MaxPool"
		
		Return
		------
		output : tuple
		    has 4 entries - (list, numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int, str)
		"""
		image       = op.inputs[0]
		
		image_shape = onnxshape_to_intlist(image.shape)

		auto_pad = None
		ceil_mode = 0
		storage_order = 0

		for attribute in op.attribute:
			if attribute.name == 'kernel_shape':
				kernel_shape = attribute.ints
			if attribute.name == 'strides':
				strides = attribute.ints
			elif attribute.name == 'pads':
				padding = attribute.ints
			elif attribute.name == 'dilations':
				dilations = attribute.ints
			elif attribute.name == 'auto_pad':
				auto_pad = attribute.s
			elif attribute.name == 'ceil_mode':
				ceil_mode = attribute.i
			elif attribute.name == 'storage_order':
				storage_order = attribute.i
		return image_shape, kernel_shape, strides, padding, dilations, auto_pad, ceil_mode, storage_order
	
	
	def nonlinearity_resources(self, op):
		"""
		This function only outputs an empty tuple, to make the code look more consistent
		
		Return
		------
		output : tuple
		    but is empty
		"""
		return ()

