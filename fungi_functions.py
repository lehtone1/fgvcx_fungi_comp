from fastai.vision import * 

def load_model_type(model_type):
    model = getattr(models, model_type)
    return model

def load_tfms(t):
    tfms_dict = {
        "t1" : None,
        "t2" : get_transforms()
    }
    tfms = tfms_dict[t]
    return tfms
    

def load_image_size(image_size_str):
    image_size = int(image_size_str)
    return image_size

def load_variables(model_name):
    variable_list = model_name.split("_")
    t = variable_list[1]
    model = load_model_type(variable_list[0])
    tfms = load_tfms(t)
    image_size = load_image_size(variable_list[2])
    return (model, tfms, t, image_size)

# Data Visualization

def get_class_number(data, class_name):
    return data.y.classes.index(class_name)

def get_class_indeces(data, class_number):
    return [index for index, value in enumerate(data.y.items) if value == class_number]

def show_images(class_name, images): 
    for image in images:
        Image.show(image, title=class_name)

def show_images_from_class(data, class_name):
    class_number = get_class_number(data, class_name)
    class_indeces = get_class_indeces(data, class_number)
    images = data.x[class_indeces]
    show_images(class_name, images)
    
    