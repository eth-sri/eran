'''
@author: Adrian Hoffmann
'''
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.framework import graph_util
import onnx





def tensorshape_to_intlist(tensorshape):
	"""
	TensorFlow has its own wrapper for shapes because some entries could be None. This function turns them into int-lists. None will become a 1.
	
	Arguments
	---------
	tensorshape : tf.TensorShape
	
	Return
	------
	output : list
	    list of ints corresponding to tensorshape
	"""
	return list(map(lambda j: 1 if j.value is None else int(j), tensorshape))


def calculate_padding(padding_str, image_shape, filter_shape, strides):
	is_valid_padding = padding_str == 'VALID'

	pad_top = 0
	pad_left = 0
	if not is_valid_padding:
		if image_shape[0] % strides[0] == 0:
			tmp = filter_shape[0] - strides[0]
			pad_along_height = max(tmp, 0)
		else:
			tmp = filter_shape[0] - (image_shape[1] % strides[0])
			pad_along_height = max(tmp, 0)
		if image_shape[1] % strides[1] == 0:
			tmp = filter_shape[1] - strides[1]
			pad_along_width = max(tmp, 0)
		else:
			tmp = filter_shape[1] - (image_shape[2] % strides[1])
			pad_along_width = max(tmp, 0)
		pad_top = int(pad_along_height / 2)

		pad_left = int(pad_along_width / 2)
	return pad_top, pad_left


class TFTranslator:
	"""
	This class is used to turn a TensorFlow model into two lists that then can be processed by an Optimizer object
	"""	
	def __init__(self, model, session = None):
		"""
		This constructor takes a reference to a TensorFlow Operation or Tensor or Keras model and then applies the two TensorFlow functions
		graph_util.convert_variables_to_constants and graph_util.remove_training_nodes to cleanse the graph of any nodes that are linked to training. This leaves us with 
		the nodes you need for inference. 
		In the resulting graph there should only be tf.Operations left that have one of the following types [Const, MatMul, Add, BiasAdd, Conv2D, Reshape, MaxPool, Placeholder, Relu, Sigmoid, Tanh]
		If the input should be a Keras model we will ignore operations with type Pack, Shape, StridedSlice, and Prod such that the Flatten layer can be used.
		
		Arguments
		---------
		model : tensorflow.Tensor or tensorflow.Operation or tensorflow.python.keras.engine.sequential.Sequential or keras.engine.sequential.Sequential
		    if tensorflow.Tensor: model.op will be treated as the output node of the TensorFlow model. Make sure that the graph only contains supported operations after applying
		                          graph_util.convert_variables_to_constants and graph_util.remove_training_nodes with [model.op.name] as output_node_names
		    if tensorflow.Operation: model will be treated as the output of the TensorFlow model. Make sure that the graph only contains supported operations after applying
		                          graph_util.convert_variables_to_constants and graph_util.remove_training_nodes with [model.op.name] as output_node_names
		    if tensorflow.python.keras.engine.sequential.Sequential: x = model.layers[-1].output.op.inputs[0].op will be treated as the output node of the Keras model. Make sure that the graph only
		                          contains supported operations after applying graph_util.convert_variables_to_constants and graph_util.remove_training_nodes with [x.name] as
		                          output_node_names
		    if keras.engine.sequential.Sequential: x = model.layers[-1].output.op.inputs[0].op will be treated as the output node of the Keras model. Make sure that the graph only
		                          contains supported operations after applying graph_util.convert_variables_to_constants and graph_util.remove_training_nodes with [x.name] as
		                          output_node_names
		session : tf.Session
		    session which contains the information about the trained variables. If None the code will take the Session from tf.get_default_session(). If you pass a keras model you don't have to
		    provide a session, this function will automatically get it.
		"""	
		output_names = None
		if issubclass(model.__class__, tf.Tensor):
			output_names = [model.op.name]
		elif issubclass(model.__class__, tf.Operation):
			output_names = [model.name]
		elif issubclass(model.__class__, Sequential):
			session      = tf.keras.backend.get_session()
			output_names = [model.layers[-1].output.op.inputs[0].op.name]
			model        = model.layers[-1].output.op
		elif issubclass(model.__class__, onnx.ModelProto):
			assert 0, 'not tensorflow model'
		else:
			import keras
			if issubclass(model.__class__, keras.engine.sequential.Sequential):
				session      = keras.backend.get_session()
				output_names = [model.layers[-1].output.op.inputs[0].op.name]
				model        = model.layers[-1].output.op
			else:
				assert 0, "ERAN can't recognize this input"
		
		if session is None:
			session = tf.get_default_session()
		
		tmp = graph_util.convert_variables_to_constants(session, model.graph.as_graph_def(), output_names)
		self.graph_def = graph_util.remove_training_nodes(tmp)	
	
	
		
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
		operation_types     = []
		operation_resources = []
		reshape_map = {}
		operations_to_be_ignored = ["Reshape", "Pack", "Shape", "StridedSlice", "Prod", "ConcatV2"]
		operations_to_be_ignored_without_reshape = ["NoOp", "Assign", "Const", "RestoreV2", "SaveV2", "IsVariableInitialized", "Identity"]

		with tf.Graph().as_default() as graph:
			with tf.Session() as sess:
				self.sess = sess
				tf.import_graph_def(self.graph_def)
				for op in graph.get_operations():
					if op.type in operations_to_be_ignored_without_reshape:
						continue
					elif op.type in operations_to_be_ignored:
						input_name  = op.inputs[0].name
						output_name = op.outputs[0].name
						kind        = op.inputs[0].op.type
						if kind in operations_to_be_ignored:
							reshape_map[output_name] = reshape_map[input_name]
						else:
							reshape_map[output_name] = input_name
						continue
			
					operation_types.append(op.type)
					input_tensor_names = []
					for inp in op.inputs:
						name = inp.name
						kind = inp.op.type
						if kind in operations_to_be_ignored:
							name = reshape_map[name]
						if kind == 'Const':
							continue
						input_tensor_names.append(name)
					in_out_info = (input_tensor_names, op.outputs[0].name, tensorshape_to_intlist(op.outputs[0].shape))
			
					if op.type == "MatMul":
						deeppoly_res = self.matmul_resources(op) + in_out_info
						deepzono_res = deeppoly_res 
						operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
					elif op.type == "Add":
						left_type  = op.inputs[0].op.type
						right_type = op.inputs[1].op.type
						if left_type == 'Const' and right_type == 'Const':
							assert 0, "we don't support the addition of two constants yet"
						elif left_type == 'Const' or right_type == 'Const':
							deeppoly_res = self.add_resources(op) + in_out_info
							deepzono_res = deeppoly_res
							operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
						else:
							operation_types[-1] = "Resadd"
							operation_resources.append({'deepzono':in_out_info, 'deeppoly':in_out_info})
					elif op.type == "BiasAdd":
						if op.inputs[1].op.type == 'Const':
							deeppoly_res = self.add_resources(op) + in_out_info
							deepzono_res = deeppoly_res
							operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
						else:
							assert 0, "this bias add doesn't meet our assumption (bias is constant)"
					elif op.type == "Conv2D":
						filters, image_shape, strides, pad_top, pad_left = self.conv2d_resources(op)
						deeppoly_res = (filters, image_shape, strides, pad_top, pad_left) + in_out_info
						deepzono_res = deeppoly_res 
						operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
					elif op.type == "MaxPool":
						image_shape, window_size, strides, pad_top, pad_left = self.maxpool_resources(op)
						deeppoly_res =  (image_shape, window_size, strides, pad_top, pad_left) + in_out_info
						deepzono_res = deeppoly_res
						operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
					elif op.type in ["Placeholder", "PlaceholderWithDefault"]:
						deeppoly_res = in_out_info
						deepzono_res = in_out_info
						operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
					elif op.type in ["Relu", "Sigmoid", "Tanh", "Softmax"]:
						deeppoly_res = self.nonlinearity_resources(op) + in_out_info
						deepzono_res = deeppoly_res
						operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
					#elif op.type == "ConcatV2":
					#	print("Concatv2")
					#	deeppoly_res = self.concat_resources(op)
					#	deepzono_res = deeppoly_res + in_out_info
					#	operation_resources.append({'deepzono':deepzono_res, 'deeppoly':deeppoly_res})
					else:
						#print("operation type1 ",in_out_info,op.inputs[0].shape,op.inputs[1].shape)
						assert 0, "Operations of type " + op.type + " are not yet supported."
		
				return operation_types, operation_resources



	def matmul_resources(self, op):
		"""
		checks which one of the direct ancestor tf.Operations is a constant and returns the underlying tensor as a numpy.ndarray inside a tuple. The matrix is manipulated in a way that it can be 
		used as the left multiplier in the matrix multiplication.
		
		Arguments
		---------
		op : tf.Operation
		    must have type "MatMul"
		
		Return 
		------
		output : tuple
		    tuple with the matrix (of type numpy.ndarray) as its only item  
		"""
		inputs = op.inputs
		left   = inputs[0]
		right  = inputs[1]
		
		if left.op.type == "Const":
			matrix = self.sess.run(left) if not op.get_attr("transpose_a") else self.sess.run(left).transpose()
		else:
			matrix = self.sess.run(right).transpose() if not op.get_attr("transpose_b") else self.sess.run(right)
		return (matrix,)
	
	
	def add_resources(self, op):
		"""
		checks which one of the direct ancestor tf.Operations is a constant and returns the underlying tensor as a numpy.ndarray inside a tuple.
		
		Arguments
		---------
		op : tf.Operation
		    must have type "Add"
		
		Return 
		------
		output : tuple
		    tuple with the addend (of type numpy.ndarray) as its only item   
		"""
		inputs = op.inputs
		left   = inputs[0]
		right  = inputs[1]
		
		if left.op.type == "Const":
			addend = self.sess.run(left)
		else:
			addend = self.sess.run(right)
		return (addend,)
		
		
	def conv2d_resources(self, op):
		"""
		Extracts the filter, the stride of the filter, and the padding from op as well as the shape of the input coming into op
		
		Arguments
		---------
		op : tf.Operation
		    must have type "Conv2D"
		
		Return 
		------
		output : tuple
		    has 4 entries (numpy.ndarray, numpy.ndarray, numpy.ndarray, str)
		"""
		inputs  = op.inputs
		image   = inputs[0]
		filters = op.inputs[1]
		
		filters     = self.sess.run(filters)
		image_shape = tensorshape_to_intlist(image.shape)[1:]
		strides     = op.get_attr('strides')[1:3]
		padding_str = op.get_attr('padding').decode('utf-8')
		pad_top, pad_left = calculate_padding(padding_str, image_shape, filters.shape, strides)
		return filters, image_shape, strides, pad_top, pad_left
	
	
	def maxpool_resources(self, op):
		"""
		Extracts the incoming image size (heigth, width, channels), the size of the maxpool window (heigth, width), and the strides of the window (heigth, width)
		
		Arguments
		---------
		op : tf.Operation
		    must have type "MaxPool"
		
		Return
		------
		output : tuple
		    has 4 entries - (list, numpy.ndarray, numpy.ndarray, str)
		"""
		image       = op.inputs[0]
		
		image_shape = tensorshape_to_intlist(image.shape)[1:]
		window_size = op.get_attr('ksize')[1:3]
		strides     = op.get_attr('strides')[1:3]
		padding_str = op.get_attr('padding').decode('utf-8')
		pad_top, pad_left = calculate_padding(padding_str, image_shape, window_size, strides)

		return image_shape, window_size, strides, pad_top, pad_left
	
	
	def nonlinearity_resources(self, op):
		"""
		This function only outputs an empty tuple, to make the code look more consistent
		
		Return
		------
		output : tuple
		    but is empty
		"""
		return ()

