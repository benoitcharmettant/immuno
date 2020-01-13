# Immuno

Directory to conduct experiments on the Immuno dataset

# Adding a new model
* 1st step: implement the model in a new file from the model package (see exemple with convnet and convnet_1)
* 2nd step: add the model to the model_manager function in models/__init__.py associated with its name 
* 3rd step: add the model's name to the list of possible choices for the model argument in the parser from experiment_manager/parser.py
