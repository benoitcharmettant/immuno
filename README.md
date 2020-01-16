# Immuno

Directory to conduct experiments on the Immuno dataset

# Adding a new model
* 1st step: implement the model in a new file from the model package (see example with convnet and convnet_1)
* 2nd step: add the model to the model_manager function in models/\_\_init\_\_.py associated with its name 
* 3rd step: add the model's name to the list of possible choices for the model argument in the parser from experiment_manager/parser.py

# Different kind of experiments

Here follows a description of all the different kind of experiments that can be conducted using this package.
You can think of it as the different kind questions that can be asked to the models. Therefor some models mights not be compatible with all the different questions.

* exp_1 : Simple classification of a given patch taken independently whether the treatment has started or not.
No distinction is made between injected tumors and control tumors. The assumption is that considering all patients have 
received Prembroluzimab, we might see an effect on every tumor.

* exp_2 : To classifications are asked here. We consider the same question as in exp_1, but we add a second class for the 
model to make a distinction between a tumor that has been injected or not (of course the treatment has to have begun).

* exp_3 : Classification taking into account the whole tumor. In this experiment a patch designate a ROI centered around
the tumor.