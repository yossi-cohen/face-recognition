import os

def get_models_path():
    path = os.path.join(
        os.path.split(
            os.path.abspath(
                os.path.join(__file__, '../..')))[-2], 'models')
    return path

def get_model_path(model_type, model):
    return os.path.join(get_models_path(), os.path.join(model_type, model))

########################################################################################
# get paths and labels for known faces.
# single image per person: file_name == person_name
# multiple images per person: put images under folder where folder_name == person_name
########################################################################################

def enum_known_faces(path):
    print('enumerating faces in:', path)
    for f in sorted(os.listdir(path)):
        full_path = os.path.join(path, f)
        if os.path.isdir(full_path):
            # multiple images per label
            label = f # label is folder name
            for f2 in os.listdir(full_path):
                image_path = os.path.join(full_path, f2)
                yield label, image_path
        else:
            # single image per label
            # label is file name without extention
            label = os.path.splitext(f)[0]
            yield label, full_path
