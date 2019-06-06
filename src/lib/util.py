import os

def get_models_path():
    return os.path.join(
        os.path.split(
            os.path.relpath(
                os.path.join(__file__, '../..')))[-2], 'models')

def get_model_path(model_type, model):
    return os.path.join(get_models_path(), os.path.join(model_type, model))
