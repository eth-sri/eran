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


from tensorflow_translator import *
from onnx_translator import *
from optimizer import *
from analyzer import *


class ERAN:
    def __init__(self, model, session=None, is_onnx = False):
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
        if is_onnx:
            translator = ONNXTranslator(model, False)
        else:
            translator = TFTranslator(model, session)
        operations, resources = translator.translate()
        self.input_shape = resources[0]["deeppoly"][2]
        self.optimizer  = Optimizer(operations, resources)
        print('This network has ' + str(self.optimizer.get_neuron_count()) + ' neurons.')
    
    
    def analyze_box(self, specLB, specUB, domain, timeout_lp, timeout_milp, use_default_heuristic,
                    output_constraints=None, lexpr_weights= None, lexpr_cst=None, lexpr_dim=None, uexpr_weights=None,
                    uexpr_cst=None, uexpr_dim=None, expr_size=0, testing = False,label=-1, prop = -1,
                    spatial_constraints=None, K=3, s=-2, timeout_final_lp=100, timeout_final_milp=100, use_milp=False,
                    complete=False, terminate_on_failure=True, partial_milp=False, max_milp_neurons=30, approx_k=True):
        """
        This function runs the analysis with the provided model and session from the constructor, the box specified by specLB and specUB is used as input. Currently we have three domains, 'deepzono',      		'refinezono' and 'deeppoly'.
        
        Arguments
        ---------
        specLB : numpy.ndarray
            ndarray with the lower bound of the input box
        specUB : numpy.ndarray
            ndarray with the upper bound of the input box
        domain : str
            either 'deepzono', 'refinezono', 'deeppoly', or 'refinepoly', decides which set of abstract transformers is used.
            
        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain isn't valid, must be 'deepzono' or 'deeppoly'"
        specLB = np.reshape(specLB, (-1,))
        specUB = np.reshape(specUB, (-1,))
        nn = layers()
        nn.specLB = specLB
        nn.specUB = specUB
        if domain == 'deepzono' or domain == 'refinezono':
            execute_list, output_info = self.optimizer.get_deepzono(nn,specLB, specUB)
            analyzer = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, output_constraints,
                                use_default_heuristic, label, prop, testing, K=K, s=s,
                                timeout_final_lp=timeout_final_lp, timeout_final_milp=timeout_final_milp,
                                use_milp=use_milp, complete=complete,
                                partial_milp=partial_milp, max_milp_neurons=max_milp_neurons)
        elif domain == 'deeppoly' or domain == 'refinepoly':
            execute_list, output_info = self.optimizer.get_deeppoly(nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size, spatial_constraints)
            analyzer = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, output_constraints,
                                use_default_heuristic, label, prop, testing, K=K, s=s,
                                timeout_final_lp=timeout_final_lp, timeout_final_milp=timeout_final_milp,
                                use_milp=use_milp, complete=complete,
                                partial_milp=partial_milp, max_milp_neurons=max_milp_neurons,
                                approx_k=approx_k)
        dominant_class, nlb, nub, failed_labels, x = analyzer.analyze(terminate_on_failure=terminate_on_failure)
        if terminate_on_failure or len(failed_labels) == 0:
            failed_labels = None # rather return nothing than an incomplete list
            
        if testing:
            return dominant_class, nn, nlb, nub, output_info
        else:
            return dominant_class, nn, nlb, nub, failed_labels, x


    def analyze_zonotope(self, zonotope, domain, timeout_lp, timeout_milp, use_default_heuristic, output_constraints=None, testing = False, prop=-1):
        """
        This function runs the analysis with the provided model and session from the constructor, the box specified by specLB and specUB is used as input. Currently we have three domains, 'deepzono',      		'refinezono' and 'deeppoly'.

        Arguments
        ---------
        original : numpy.ndarray
            ndarray with the original input
        zonotope : numpy.ndarray
            ndarray with the zonotope
        domain : str
            either 'deepzono', 'refinezono', 'deeppoly' or 'refinepoly', decides which set of abstract transformers is used.

        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        assert domain in ['deepzono', 'refinezono'], "domain isn't valid, must be 'deepzono' or 'refinezono'"
        nn = layers()
        nn.zonotope = zonotope
        if domain == 'deepzono' or domain == 'refinezono':
            execute_list, output_info   = self.optimizer.get_deepzono(nn, zonotope)
            analyzer       = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, testing, prop=prop)
        elif domain == 'deeppoly' or domain == 'refinepoly':
            assert 0
            #execute_list   = self.optimizer.get_deeppoly(original, zonotope, True)
            #analyzer       = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, specnumber, use_default_heuristic)
        dominant_class, nlb, nub, failed_labels, x = analyzer.analyze()
        return dominant_class, nn, nlb, nub, failed_labels, x

