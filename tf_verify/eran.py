'''
@author: Adrian Hoffmann
'''

from tensorflow_translator import *
from optimizer import *
from analyzer import *


class ERAN:
    def __init__(self, model, session=None):
        """
        This constructor takes a reference to a TensorFlow Operation, TensorFlow Tensor, or Keras model. The two TensorFlow functions graph_util.convert_variables_to_constants and 
        graph_util.remove_training_nodes will be applied to the graph to cleanse it of any nodes that are linked to training.
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
            session which contains the information about the trained variables. If session is None the code will take the Session from tf.get_default_session(). If you pass a keras model you don't 
            have to provide a session, this function will automatically get it.
        """
        translator = TFTranslator(model, session)
        operations, resources = translator.translate()
        self.optimizer  = Optimizer(operations, resources)
    
    
    def analyze_box(self, specLB, specUB, domain, timeout_lp, timeout_milp, use_area_heuristic, specnumber=0):
        """
        This function runs the analysis with the provided model and session from the constructor, the box specified by specLB and specUB is used as input. Currently we have three domains, 'deepzono',      		'refinezono' and 'deeppoly'.
        
        Arguments
        ---------
        specLB : numpy.ndarray
            ndarray with the lower bound of the input box
        specUB : numpy.ndarray
            ndarray with the upper bound of the input box
        domain : str
            either 'deepzono', 'refinezono' or 'deeppoly', decides which set of abstract transformers is used.
            
        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        assert domain in ['deepzono', 'refinezono', 'deeppoly'], "domain isn't valid, must be 'deepzono' or 'deeppoly'"
        specLB = np.reshape(specLB, (-1,))
        specUB = np.reshape(specUB, (-1,))
        nn = layers()
        nn.specLB = specLB
        nn.specUB = specUB
        if domain == 'deepzono' or domain == 'refinezono':
            execute_list   = self.optimizer.get_deepzono(nn,specLB, specUB)
            analyzer       = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, specnumber, use_area_heuristic)
        elif domain == 'deeppoly':
            execute_list   = self.optimizer.get_deeppoly(specLB, specUB)
            analyzer       = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, specnumber, use_area_heuristic)
        dominant_class, nlb, nub = analyzer.analyze()
        return dominant_class, nn, nlb, nub
        
