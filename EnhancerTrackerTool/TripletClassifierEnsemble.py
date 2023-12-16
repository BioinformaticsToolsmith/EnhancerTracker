import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
from tensorflow import keras
import os
import nets
import random
from loaders import PremadeTripletClassifierSequence
import numpy as np
import sys 
import pickle as pk
import gc
# from tensorflow.keras.utils import get_custom_objects
# get_custom_objects().update({'Specificity': nets.Specificity})

def exponential_decay_fn(epoch, lr):
    '''
    This function decreases the learning rate according to the epoch
    '''
    return lr*0.1**(1/100)

class Model:
    """
    Holds the information about a trained model such as what it was trained with (parameters).
    """
    def __init__(self, model_id):
        """
        model_id: Unique ID for the model
        """
        
        self.model_id = model_id
        
        # Intializing all class variables to None
        self.codings_size = None
        self.exp_filter_num = None
        self.exp_filter_1d_size = None
        self.blocks = None
        self.learning_rate = None
        self.allow_reverse = None
        self.metric = None
        
        self.training_average = None
        self.validation_average = None
        self.vote_weight = None
        
        # Some method will fail if self.is_set is False
        self.is_set = False
        
        # The indexes of the metrics when training the model; i.e., when evaluate is called on a tensorflow model, what index corresponds to what metric.
        self.vote_dict = {"accuracy":0, "specificity":1, "recall":2, "precision":3, "f1_score":4, "all":5}
        
    def set_values(self, c, en, es, b, lr, ar, ls, m, ta, va):
        """
        Sets the training information
        """
        
        self.is_set = True
        
        self.codings_size = c
        self.exp_filter_num = en
        self.exp_filter_1d_size = es
        self.blocks = b              # Number of convolutional blocks + 1
        self.learning_rate = lr
        self.allow_reverse = ar      # reverse_complement
        self.loss = ls
        self.metric = m              # What metric was used for early stopping
        self.training_average = ta   # Evaluation of metrics on training
        self.validation_average = va # Evaluation of metrics on validation
        
    def get_id(self):
        """
        returns the unique ID of the model
        """
        return self.model_id
    
    def set_vote(self, metric):
        """
        Sets the weight of the vote for this model if unequal vote is being used
        """
        assert self.is_set, "The model has not been trained yet!"
        assert metric in self.vote_dict, f"The metric given is not in the dictionary: {metric} not in {self.vote_dict.keys()}"
        
        if metric == "all":
            self.vote_weight = sum(self.training_average) / len(self.training_average) 
        else:
            self.vote_weight = self.training_average[self.vote_dict[metric]]
        
    def __str__(self):
        """
        Returns a string containing all of the parameters used to train the model as well as the votes and evaluation of the model
        """
        assert self.is_set, "The model has not been trained yet!"
        
        labels = ["Codings Size", "Filter Number", "Filter 1D Size", "Conv Blocks", "Learning Rate", "Allow Reverse", "Loss", "Monitor Metric", "Training Average", "Validation Average", "Vote Weight"]
        values = [self.codings_size, self.exp_filter_num, self.exp_filter_1d_size, self.blocks, self.learning_rate, self.allow_reverse, self.loss, self.metric, self.training_average, self.validation_average, self.vote_weight]
        
        return '--------------\n' + '\n'.join([f"{labels[i]}: {values[i]}" for i in range(len(labels))])
    
    def get_path(self, directory):
        """
        Given the directory, return a path with the unique model ID at the end
        directory: directory that contains the model
        """
        return f'{directory}/{self.model_id}'
    
    def load_model(self, directory, custom_objects = None):
        """
        Loads the tensorflow model into memory
        directory     : directory that contains the model
        custom_objects: customly defined metrics; tensorflow model will not load without these
        """
        assert self.is_set, "The model has not been trained yet!"
        self.model =  keras.models.load_model(self.get_path(directory), custom_objects = custom_objects)



    
    
    
    

class TripletClassifierEnsemble:
    '''
    An ensemble classifier that builds a diverse grouping of classifiers (randomly chosen parameters) and gets their consensus for predicting.
    '''
    
    def __init__(self, out_dir, shape):
        '''
        out_dir: where the models are saved and loaded from as well as where the models' information is stored
        shape  : shape of the input data; should be three dimensions
        '''
        
        self.model_count = 0
        
        self.out_dir = out_dir
        
        # Will contain the meta-information for each model
        self.model_list = []
        
        # List of metrics that will be used; first index of each set is the class/function of the metric, second is the name, third is if it is keras-defined (True) or custom-defined (False)
        self.metric_list = [(tf.keras.metrics.Accuracy(), "accuracy", True), (nets.crm_specificity, "crm_specificity", False), (tf.keras.metrics.Recall(), "recall", True), (tf.keras.metrics.Precision(), "precision", True), (nets.crm_f1_score, "crm_f1_score", False)]

        # Shape of the input data
        self.d1, self.d2, self.d3 = shape
                           
        assert os.path.exists(out_dir)
        
        
        # Custom objects for loading and saving models 
        self.custom_objects = {"crm_specificity": nets.crm_specificity, "crm_f1_score": nets.crm_f1_score}
           
    def average_metrics(self, seq, model, average_count):
        """
        Given a sequence and a tensorflow model, evaluate the model multiple times and return the averages of each metric.
        seq          : sequence to evaluate
        model        : model to use for evaluation
        average_count: number of times to evaluate and average over.
        """
        l = []
        for j in range(average_count):
            l.append(model.evaluate(seq)[1:])

        # Averages each metric
        am = [sum(metric)/len(metric) for metric in zip(*l)]
        return am
            
    def fit(self, model_count, train, reverse_train, train_triplet_sim, train_triplet_dis, valid, reverse_valid, valid_triplet_sim, valid_triplet_dis, param_dict = None, average_count = 3, epochs = 500, start_from = 0, is_load = False, vote_metric = "all"):
        '''
        Creates a model for the purpose of fitting and saves it.
        This model will have random parameters. 
        model_count:       the number of models for the ensemble to use
        train:             the one-hot encodings of the train sequences
        reverse_train:     the one-hot encodings of the reverse complemented train sequences
        train_triplet_sim: the path to the numpy file containing the indexes for training triplets with Anchor, Positive, and Similars
        train_triplet_dis: the path to the numpy file containing the indexes for training triplets with Anchor, Positive, and Negative
        valid:             the one-hot encodings of the valid sequences
        reverse_valid:     the one-hot encodings of the reverse complemented valid sequences
        valid_triplet_sim: the path to the numpy file containing the indexes for validation triplets with Anchor, Positive, and Similars
        valid_triplet_dis: the path to the numpy file containing the indexes for validation triplets with Anchor, Positive, and Negative  
        param_dict:        should contain keys for codings size, number of filters, size of filters, number of convolutional blocks, learning rate, and what metric to monitor. All but the number of filters and metric should have a tuple with the min and max range to randomly choose from. The # of filters and metric should have a list of valid values to choose from.
        average_count:     Number of times to evaluate a model on the training and validation data
        epochs: Max number of epochs to train each model on
        start_from: Start training at this model count; typically if used for when the code crashes and the fit needs to be ran again
        is_load: Used for when the code crashes. If true, then load in the already trained models. If false, don't. Use false if training over multiple machines.
        vote_metric: which metric should a model's vote count torwards?
        '''
        
        assert model_count > 0, f"Model count must be greater than 0! Currently it is {model_count}"
        self.model_count = model_count
        
        assert average_count > 0, f"Average count must be greater than 0! Currently it is {average_count}"
        assert epochs > 0, f"Number of epochs must be greater than 0! Currently it is {epochs}"
        assert start_from >= 0, f"The starting model can not be less than 0! Currently it is {start_from}"
        
        valid_metrics = ["accuracy", "specificity", "recall", "precision", "f1_score", "all"]
        assert vote_metric in valid_metrics, f"vote_metric is not a valid metric! Valid metrics are {valid_metrics}"

        # Parameters to randomize when training
        param_dict = param_dict if param_dict is not None else {
            "codings":(25,100),
            "exp_filter_num":[4, 8, 16],
            "exp_filter_1d_size":(3,11),
            "learning_rate":(0.05,1.0),
            "allow_reverse": [True, False],
            "loss": ["mse", "binary_crossentropy"],
            "metric":['val_accuracy', 'val_precision', 'val_crm_f1_score']
        }
        
        # Load the models and start from the value in start_from
        if start_from != 0 and is_load:
            self.load_model_info()
        
        # Otherwise, create the model list and start training
        else:
            # Create an empty list of None depending on the model_count set 
            self.model_list = [None] * self.model_count
            

        for i in range(start_from, self.model_count):
            
            # Choose different parameters through randomization
            codings_size = random.randint(*param_dict["codings"])
            exp_filter_num = random.choice(param_dict["exp_filter_num"])
            exp_filter_1d_size = random.randint(*param_dict["exp_filter_1d_size"])
            block = 4
            learning_rate = random.uniform(*param_dict["learning_rate"])
            allow_reverse = random.choice(param_dict["allow_reverse"])
            loss = random.choice(param_dict["loss"])
            metric = random.choice(param_dict["metric"])
            
            # Create the dataset sequences
            train_seq = PremadeTripletClassifierSequence(train, train_triplet_sim, train_triplet_dis, batch_size = 1024, reverse_x_in = reverse_train if allow_reverse else None)
            valid_seq = PremadeTripletClassifierSequence(valid, valid_triplet_sim, valid_triplet_dis, batch_size = 1024, reverse_x_in = reverse_valid if allow_reverse else None)
            
            # Creating our model and compiling 
            model = nets.make_conv_classifier_blocks(codings_size, (self.d1, self.d2, self.d3), exp_filter_1d_size, exp_filter_num, block)

            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

            model.compile(loss=loss, metrics=['accuracy', nets.crm_specificity, tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision'), nets.crm_f1_score], optimizer=opt)  
            early_stopping = keras.callbacks.EarlyStopping(patience=20, min_delta=1/100000, restore_best_weights=True, monitor=metric, start_from_epoch=10, mode='max') # start_from_epoch=20
            lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

            
            print(f"Model {i} using:\nCodings_size:\t{codings_size}\nexp_filter_num:\t{exp_filter_num}\nlearning_rate:\t{learning_rate}\nallow_reverse:\t{allow_reverse}\nloss:\t{loss}\nmetric:\t{metric}")
            
            # Fitting and saving our model
            model.fit(train_seq, epochs=epochs, validation_data=valid_seq, workers=26, callbacks=[early_stopping, lr_scheduler])
            self.model_list[i] = Model(f"model_{i + 1}")
        
            model.save(self.model_list[i].get_path(self.out_dir))
                     
            
            # Evaluating model
            training_average = self.average_metrics(train_seq, model, average_count)
            validation_average = self.average_metrics(valid_seq, model, average_count)
            
            # Setting parameters chosen for model
            # self.model_list[i] = Model(f"model_{i}")
            self.model_list[i].set_values(codings_size, exp_filter_num, exp_filter_1d_size, block, learning_rate, allow_reverse, loss, metric, training_average, validation_average)
            
            self.model_list[i].set_vote(vote_metric)
            
            # with open(f'{self.out_dir}/{self.model_list[i].get_path(self.out_dir)}/model_info.pickle', 'wb') as f:
            with open(f'{self.model_list[i].get_path(self.out_dir)}/model_info.pickle', 'wb') as f:
                pk.dump(self.model_list[i], f)
                
            gc.collect()
                
                
    def calculate_metrics(self, prediction, y):
        """
        Given a prediction and the ground truth, evaluate using accuracy, specificity, recall, precision, and F1 score
        prediction: predicted labels
        y         : true labels
        """

        # accuracy = nets.crm_accuracy(y, prediction)
        # specificity = nets.crm_specificity(y, prediction)
        # recall = nets.crm_recall(y, prediction)
        # precision = nets.crm_precision(y, prediction)
        # f1 = nets.crm_f1_score(y, prediction)
        prediction = tf.cast(tf.math.round(prediction), tf.int32)
        y = tf.cast(y, tf.int32)
        
        c = 0.0
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0
        for i in range(len(y)):
            if y[i] == 1:
                if prediction[i] == 1:
                    tp += 1
                    c += 1
                else:
                    fn += 1
            else:
                if prediction[i] == 1:
                    fp += 1
                else:
                    tn += 1
                    c += 1
                    
            
        accuracy = c / (len(y) + sys.float_info.epsilon)
        specificity = tn / (tn + fp + sys.float_info.epsilon)
        recall = tp / (tp + fn + sys.float_info.epsilon)
        precision = tp / (tp + fp + sys.float_info.epsilon)
        f1 = (2 * recall * precision) / (recall + precision + sys.float_info.epsilon)
        
        print("Accuracy:", accuracy)
        print("Specificity:", specificity)
        print("Recall: ", recall )
        print("Precision: ", precision )
        print("F1: ",  f1)
        
        r = [accuracy, specificity, recall, precision, f1]
        
        return r
                    
    def predict(self, x, use_count = -1, threshold = 0.55, is_merit = False, is_loaded = True, verbose = 1, already_predicted = False):
        '''
        Predicts on sequence using multiple models
        x        : the sequence to predict on; if already_predicted is True, then this is a list of predictions already made
        use_count: the number of models to use; if -1, use all. If greater than total number of models, uses all models
        threshold: threshold to determine if consensus prediction should be 1
        is_merit : is the vote based on the weight of the model? If true, then yes, otherwise, uses equal weight for all models
        is_loaded: if the models are loaded into memory; faster if true, very slow if not.
        verbose  : how verbose should the prediction be; default of 1 so will show the batch for every model
        already_predicted: is x a list of already made predictions?
        '''
        seq_len = len(x) if not already_predicted else len(x[0])
        
        if already_predicted:
            use_count = len(x)
        elif use_count == -1 or use_count > self.model_count:
            use_count = self.model_count
        
        use_list = self.model_list[:use_count]       # Models to use, based on use_count
        vote_array = np.zeros((seq_len, use_count))   # Holds the votes of every model
        consensus = np.zeros(len(vote_array), dtype = np.ubyte)        # Holds the consensus agreement
        confidence_array = np.zeros(len(vote_array), dtype = np.float16) # Holds the confidence of the consensus agreement
        
        if already_predicted:
            for i in range(len(x)):
                vote_array[:, i] = x[i]
        else:
            
            # Determines how the models are used to predicted based on if they are loaded into memory or not.
            if is_loaded:
                def predict_model(m, x):
                    return m.model.predict(x, verbose = verbose).reshape(-1)
            else:
                def predict_model(m, x):
                    return keras.models.load_model(m.get_path(self.out_dir), custom_objects = self.custom_objects).predict(x, verbose = verbose).reshape(-1)
            # else:
            #     def predict_model(m, x):
            #         return tf.cast(tf.math.round(keras.models.load_model(m.get_path(self.out_dir), custom_objects = self.custom_objects).predict(x, verbose = verbose).reshape(-1)), tf.uint8)

            for i, m in enumerate(use_list):
                #p = tf.cast(tf.math.round(predict_model(m, x)), tf.uint8)
                p = predict_model(m, x)
                assert p.shape[0] == len(x)
                vote_array[:, i] = p
                gc.collect()
            
        # If there are more than 1 models
        if use_count > 1:
            
            # Unequal vote; using the voting weight of a model
            if is_merit:       
                total_weight = sum([m.vote_weight for m in use_list])
                for row_index in range(len(vote_array)):
                    c = 0
                    # divisor = 0
                    for j, vote in enumerate(vote_array[row_index, :]): 
                        c += (vote * use_list[j].vote_weight) if vote >= 0.5 else 0


                    confidence = c/total_weight # weighted average
                    confidence_array[row_index] = confidence

                    if confidence >= threshold:
                        consensus[row_index] = 1
                        
                
                
            # Equal vote; every model has the same vote weight
            else:
                for row_index in range(len(vote_array)):
                    c = 0
                    for vote in vote_array[row_index, :]: 
                        c += vote if vote >= 0.5 else 0.0
                        
                    confidence = c / use_count # average
                    confidence_array[row_index] = confidence

                    if confidence >= threshold: 
                        consensus[row_index] = 1


        # Otherwise, just use 1 model's prediction
        else:
            consensus[:] = np.round(vote_array[:, 0]).astype(np.ubyte)
            confidence_array = [1] * len(confidence_array)
            
        del vote_array
        del use_list
        gc.collect()
        return consensus, confidence_array 
                              
    def evaluate(self, x, y, use_count = -1, threshold = 0.55, is_merit = False, is_loaded = True, verbose = 1, already_predicted = False):
        '''
        Evaluates the ensemble
        x                : the sequence to predict on
        y                : the true labels to compare to
        use_count        : the number of models to use; if -1, use all. If greater than total number of models, uses all models
        threshold        : threshold to determine if consensus prediction should be 1
        is_merit         : is the vote based on the weight of the model? If true, then yes, otherwise, uses equal weight for all models
        is_loaded        : if the models are loaded into memory; faster if true, very slow if not.
        verbose          : how verbose should the prediction be; default of 1 so will show the batch for every model
        already_predicted: is x a list of already made predictions?
        '''
        prediction, confidence = self.predict(x, use_count = use_count, threshold = threshold, is_merit = is_merit, verbose = verbose, is_loaded = is_loaded, already_predicted = already_predicted)
               
        return self.calculate_metrics(prediction, y), confidence
    
#     def predict_from(self, prediction_list, threshold = 0.55, is_merit = False):
        
#         use_count = len(prediction_list)
#         vote_array = np.zeros((len(prediction_list[0]), use_count))
#         consensus = np.zeros(len(vote_array))
#         confidence_array = np.zeros(len(vote_array))
#         if use_count > 1:
            
#             for i in range(use_count):
#                 vote_array[:, i] = tf.math.round(prediction_list[i])
                
                
#             if is_merit:

#                 # Creating our consensus from the models
#                 consensus = np.zeros(len(vote_array))
#                 total_weight = sum([m.vote_weight for m in self.model_list[:use_count]])
#                 for instance in range(len(vote_array)):
#                     c = 0
#                     for j, vote in enumerate(vote_array[instance, :]): 
#                         c += (vote * self.model_list[j].vote_weight)
                    
#                     confidence = c/total_weight
#                     confidence_array[instance] = confidence
                    
#                     if confidence >= threshold:
#                         consensus[instance] = 1
                    
#             else:

#                 # Creating our consensus from the models
#                 for instance in range(len(vote_array)):
#                     c = 0
#                     for vote in vote_array[instance, :]:  # corrected here
#                         c += vote
                    
#                     confidence =  c/use_count
#                     confidence_array[instance] = confidence
                    
#                     if confidence >= threshold:  # corrected here
#                         consensus[instance] = 1
                        
#         # Otherwise, just use 1 model's prediction
#         elif use_count == 1:
#             consensus = tf.math.round(prediction_list[0])
#             confidence_array = [-1] * len(confidence_array) 
            
#         else:
#             raise RuntimeError(f"Use count cannot be zero! {use_count}")

#         return consensus, confidence_array
    
#     def evaluate_from(self, prediction_list, y, threshold = 0.55, is_merit = False, is_loaded = False):
#         prediction, confidence = self.predict_from(prediction_list, threshold = threshold, is_merit = is_merit)
        
#         return self.calculate_metrics(prediction, y), confidence
    
    def predict_raw(self, x, i, is_loaded = False):
        """
        Gives the prediction of just 1 model
        x        : the sequence to predict on
        i        : the index of the model to use
        is_loaded: is the model being used already loaded?
        """
        assert i < len(self.model_list), f"Index out of bounds: {i} in {len(self.model_list)}"
        model = keras.models.load_model(self.model_list[i].get_path(self.out_dir), custom_objects = self.custom_objects) if not is_loaded else self.model_list[i].model
        return model.predict(x).reshape(-1)
        
    def load_model_info(self, path = None):
        '''
        Loads pickled information about a model. Not the tensorflow models themselves!
        Sets model_count to the maximum depending on the largest model file 
        path: directory with the models
        '''
        if path is None:
            path = self.out_dir
        else:
            assert os.path.exists(path), f'{path} does not exist!'
            assert os.path.isdir(path), f'{path} is not a directory!'
            
        dir_list = [f"{path}/{x}" for x in os.listdir(path)]
        hold_list = []
        count_set = set()
        for p in dir_list:
            
            if os.path.isdir(p):
                c = int(os.path.basename(p).split('_')[1])
                self.model_count = max(self.model_count, c)
                hold_list.append((p, c))
                assert c not in count_set, f'Model count {c} has already been used!'
                count_set.add(c)
            
        self.model_list = [None] * self.model_count
        for p, c in hold_list:
            with open(f'{p}/model_info.pickle', 'rb') as f:
                self.model_list[c - 1] = pk.load(f)
            
            
        
        # with open(f'{path}/model_info.pickle', 'rb') as input:
        #     self.model_list = pk.load(input)
            
            
        
    def get_model_count(self):
        '''
        Gets the count of a model.
        '''
        return self.model_count
    
    def get_model_info(self, model_num = None):
        '''
        Retrieves a string of the information of the model to see parameters used. 
        '''
        if model_num is None:
            return '\n'.join([str(model) for model in self.model_list])
        else:
            return str(self.model_list[model_num])
    
    def get_model(self, model_num):
        '''
        returns a model.
        '''
        return keras.models.load_model(self.model_list[model_num].get_path(self.out_dir), custom_objects = self.custom_objects)
    
    def load_models(self, use_count = -1):
        '''
        Loads up to use_count models
        use_count: load up to this number of models into memory; if -1 or greater than number of models, load all
        '''
        use_count = self.model_count if use_count == -1 or use_count > self.model_count else use_count
        for i in range(use_count):
            sys.stdout.write(f"\rLoading models {i+1}/{use_count}")
            sys.stdout.flush()
            self.model_list[i].load_model(self.out_dir, custom_objects = self.custom_objects)

        print()
        
    def set_model_votes(self, metric):
        '''
        Sets a model's metric 
        '''
        valid_metrics = ["accuracy", "specificity", "recall", "precision", "f1_score", "all"]
        assert metric in valid_metrics, f"vote_metric is not a valid metric! Valid metrics are {valid_metrics}"    
        
        for i in range(len(self.model_list)):
            if self.model_list[i] is not None:
                self.model_list[i].set_vote(metric)

                with open(f'{self.model_list[i].get_path(self.out_dir)}/model_info.pickle', 'wb') as f:
                    pk.dump(self.model_list[i], f)

        # with open(f'{self.out_dir}/model_info.pickle', 'wb') as f:
        #     pk.dump(self.model_list, f)
