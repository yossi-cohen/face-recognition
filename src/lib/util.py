import os

def get_models_path():
    path = os.path.join(
        os.path.split(
            os.path.abspath(
                os.path.join(__file__, '../..')))[-2], 'models')
    return path

def get_model_path(model_type, model):
    return os.path.join(get_models_path(), os.path.join(model_type, model))
