import json
import numpy as np
from keras.models import model_from_json
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image

def load_questions(questions_path) :
    with open(questions_path) as data_file:
        data = json.load(data_file)
    return data['questions']

def get_quest_features(question, path):
    fname = str(question['question_id']) + '.npy'
    return np.load(path + fname)

def get_im_features(question, path):
    fname = str(question['image_id']) + '.npy'
    return np.load(path + fname)

def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    ''' Given the VQA model and its weights, compiles and returns the model '''

    # thanks the keras function for loading a model from JSON, this becomes
    # very easy to understand and work. Alternative would be to load model
    # from binary like cPickle but then model would be obfuscated to users
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

def retrieve_most_probable_answers(y_output, labelencoder, nb_answers = 5):
    labels_to_keep = np.argsort(y_output)[0,-nb_answers:]
    save = []
    for label in labels_to_keep:
        save.append({'answer' : labelencoder[label], 'proba' : round(100 * y_output[0, label], 2)})
    sol = pd.DataFrame.from_records(save)
    sol.sort_values(by = 'proba', ascending = False, inplace = True)
    return sol

def plot_question_im_answ(question, y_output, labelencoder, impath = '../../data/val2014/', nb_answers = 5):
    df = retrieve_most_probable_answers(y_output, labelencoder, nb_answers)
    df.sort_values(by = 'proba', ascending=True, inplace=True)
    img_id = question['image_id']
    img_name = 'COCO_val2014_000000{}.jpg'.format(str(img_id))
    img = image.load_img(impath + img_name)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.barh(np.arange(df.shape[0]), df.proba, align='center', alpha=0.4)
    #ax1.set_yticks(np.arange(df.shape[0]), df.answer)
    ax1.set_xlabel('proba')
    ax1.set_title(str(question['question']))
    ax2.imshow(img)
    plt.sca(ax1)
    plt.yticks(range(nb_answers), df.answer)
    plt.show()
