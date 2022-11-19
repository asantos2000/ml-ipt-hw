import tensorflow as tf
import seaborn as sns
import random
import pprint
import plotly.graph_objs as go
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras import Model
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_files
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm
from skimage.feature import hog
from plotly.subplots import make_subplots
from plotly.offline import iplot
from plotly import tools
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.datasets import fashion_mnist
import pickle


pp = pprint.PrettyPrinter(indent=4)

DB_DIR = "db"
CLASS_NAMES = ['Camiseta', 'Calça', 'Pulôver', 'Vestido', 'Casaco',
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']
AMOSTRAS_GRID = 36

# specify Classes / Labels
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               #'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
labels = dict([cl for cl in enumerate(CLASS_NAMES)])


def load_mnist_dataset():
    (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
    return train_X, train_Y, test_X, test_Y


def exibe_bitmap_primeira_imagem(train_X):
    index = random.randrange(0, len(train_X))
    plt.figure()
    plt.imshow(train_X[0], cmap="Greys")
    plt.colorbar()
    plt.grid(False)
    plt.show()


def plota_barra(y, loc):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5

    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]
    counts = counts[sorted_index]
    ylabel_text = 'count'

    xtemp = np.arange(len(unique))
    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, CLASS_NAMES, rotation=45)
    plt.xlabel('classes')
    plt.ylabel(ylabel_text)


def plota_barras_treinamento_e_teste(y_train, y_test):
    plota_barra(y_train, loc='left')
    plota_barra(y_test, loc='right')
    plt.suptitle('Quantidade relativa de imagens para cada base',)
    plt.legend([
        'treinamento ({0} imagens)'.format(len(y_train)),
        'teste ({0} imagens)'.format(len(y_test))
    ])


def exibe_grade_imagens(tam_amostras, imagens, labels, must_reshape=True):
    plt.figure(figsize=(10, 10))
    for i in range(tam_amostras):
        plt.subplot(6, 6, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if must_reshape:
            img = imagens[i].reshape((28, 28))
        else:
            img = imagens[i]
        plt.imshow(img, cmap='gray')
        label_index = int(labels[i])
        plt.title(CLASS_NAMES[label_index])
    plt.show()

# HOG


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def local_hog(X):
            return hog(X.reshape(28, 28),
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try:  # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])


def gera_amostras_hog(qtde_amostras, source):
    feat_hog = [None] * qtde_amostras
    img_hog = [None] * qtde_amostras
    for i in range(qtde_amostras):
        orig_hog, img = hog(
            source[i].reshape((28, 28)), pixels_per_cell=(14, 14),
            cells_per_block=(2, 2),
            orientations=9,
            visualize=True,
            block_norm='L2-Hys'
        )
        feat_hog[i] = orig_hog
        img_hog[i] = img
    return feat_hog, img_hog


def comparativo_imagem_hog(img, img_hog):
    img = img.reshape((28, 28))
    print('Pixels da imagem original: ', img.shape[0] * img.shape[1])
    print('Características HOG: ', img_hog.shape[0])


def transformadores():
    scalify = StandardScaler()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm='L2-Hys'
    )
    return scalify, hogify


def gera_hog_base(source):
    scalify, hogify = transformadores()
    source_hog = hogify.fit_transform(source)
    return scalify.fit_transform(source_hog)


def define_model():
    return SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)


def monta_pipeline_hog():
    scalify, hogify = transformadores()
    model = define_model()
    return Pipeline([
        ('hogify', hogify),
        ('scalify', scalify),
        ('classify', model)
    ])


def monta_pipeline():
    scalify, _ = transformadores()
    model = define_model()
    return Pipeline([
        ('scalify', scalify),
        ('classify', model)
    ])


def treina_com_hog(x, y):
    HOG_pipeline = monta_pipeline_hog()
    return HOG_pipeline.fit(x, y)


def treina_sem_hog(x, y):
    pipeline = monta_pipeline_hog()
    return pipeline.fit(x, y)


def treina_com_hog_otimizado(x, y):
    param_grid = [
        {
            'hogify__orientations': [8, 9],
            'hogify__cells_per_block': [(2, 2), (3, 3)],
            'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)]
        },
        {
            'hogify__orientations': [8],
            'hogify__cells_per_block': [(3, 3)],
            'hogify__pixels_per_cell': [(8, 8)],
            'classify': [
                define_model(),
                svm.SVC(kernel='linear')
            ]
        }
    ]
    HOG_pipeline = monta_pipeline_hog()
    grid_search = GridSearchCV(HOG_pipeline,
                               param_grid,
                               cv=3,
                               n_jobs=-1,
                               scoring='accuracy',
                               verbose=1,
                               return_train_score=True)

    return grid_search.fit(x, y)


def exibe_percentual_de_acerto(predicoes, y_test):
    print('Percentual de acerto: ', 100 *
          np.sum(predicoes == y_test) / len(y_test))


def plot_confusion_matrix(true_Y, pred_Y):
    ConfusionMatrixDisplay.from_predictions(true_Y,
                                            pred_Y,
                                            display_labels=CLASS_NAMES,
                                            xticks_rotation="45")
    plt.show()


def show_confusion_matrix(true_Y, pred_Y):
    plot_confusion_matrix(true_Y, np.argmax(pred_Y, axis=1))


def salva_modelo(model, nome):
    joblib.dump(model, "models/{0}.pkl".format(nome))


def le_modelo(nome):
    try:
        return joblib.load(f"models/{nome}.pkl")
    except:
        return None


def salva_historico(historico, nome):
    with open(f"models/{nome}.pkl", 'wb') as file:
        pickle.dump(historico, file)


def le_historico(nome):
    with open(f"models/{nome}.pkl", 'rb') as file:
        return pickle.load(file)


def exibe_otimizacao(model):
    print("Melhores parâmetros:")
    pp.pprint(model.best_params_)
    print("Melhor pontuação (score):", model.best_score_)


def formata_imagem(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
    return img


def exibe_e_retorna_imagens_para_predizer():
    filename = random.choice(os.listdir("../Images"))
    img = formata_imagem(os.path.join("../Images", filename))
    plt.title(filename)
    plt.imshow(img)
    plt.show()
    img = img.flatten()
    return np.asarray([img])

# Load extra dataset for test


def convert_image_to_array(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
    img = img.flatten()
    return np.asarray([img])


def load_extra_dataset(dir="input/test_images"):
    data = load_files(dir, load_content=False)

    data_X = []
    data_Y = data["target"]

    for filename in data["filenames"]:
        try:
            file_arr = convert_image_to_array(filename)
            file_arr_28 = file_arr[0].reshape(28, 28)
            data_X.append(file_arr_28)
        except Exception as error:
            print(f"Error loading {filename}. Exception: {error}")

    return np.array(data_X), data_Y


def get_classes_distribution(data):
    # Get the count for each label
    label_counts = data[0].value_counts()

    # Get total number of samples
    total_samples = len(data)

    # Count the number of items in each class
    for i in range(len(label_counts)):
        label = labels[label_counts.index[i]]
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print(f"{label:<20s}:   {count} or {percent:.1f}%")


def adjust_data_for_transfer_learning(dataset, size):

    dim = (size, size)

    # resize and convert grayscale to rgb channels
    def to_rgb(img):
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
        return img_rgb

    rgb_list = []

    for i in range(len(dataset)):
        rgb = to_rgb(dataset[i])
        rgb_list.append(rgb)

    rgb_arr = np.stack([rgb_list], axis=4)
    dataset = np.squeeze(rgb_arr, axis=4)

    return dataset


def get_pre_trained_model(ModelClass, weights, freeze_num_layers, image_size, channels):
    pre_trained = ModelClass(input_shape=(image_size, image_size, channels),
                             include_top=False,
                             weights=weights)

    ModelLayersClass = pre_trained.layers

    for layer in range(len(ModelLayersClass)-freeze_num_layers):
        ModelLayersClass[layer].trainable = False

    return pre_trained


def adj_model_DenseNet169(image_size, channels):
    # Get pre trained DenseNet169
    pre_trained_model_DenseNet169 = get_pre_trained_model(
        DenseNet169, None, 250, image_size, channels)

    print(
        f"Number of layer DenseNet169: {len(pre_trained_model_DenseNet169.layers)}")

    # Configure model
    x = Flatten()(pre_trained_model_DenseNet169.output)

    # Fully Connection Layers

    # FC1
    x = Dense(1024, activation="relu")(x)

    # Dropout to avoid overfitting effect
    x = Dropout(0.4)(x)

    # FC2
    x = Dense(1024, activation="relu")(x)

    # FC3
    x = Dense(1024, activation="relu")(x)

    # Dropout to avoid overfitting effect
    x = Dropout(0.2)(x)

    # FC4
    x = Dense(512, activation="relu")(x)

    # FC5
    x = Dense(512, activation="relu")(x)

    # Dropout to avoid overfitting effect
    x = Dropout(0.2)(x)

    # FC6
    x = Dense(256, activation="relu")(x)

    # FC7
    x = Dense(256, activation="relu")(x)

    # Dropout to avoid overfitting effect
    x = Dropout(0.2)(x)

    # FC8
    x = Dense(128, activation="relu")(x)

    # output layer
    x = Dense(10, activation="softmax")(x)

    # concatenation layers
    model_DenseNet169 = Model(pre_trained_model_DenseNet169.input, x)

    # Compile model
    model_DenseNet169.compile(optimizer="adam",
                              loss="categorical_crossentropy",
                              metrics=['accuracy'])

    return model_DenseNet169


def adj_model_ResNet152V2(image_size, channels):
    # Get pre trained ResNet152V2
    pre_trained_model_ResNet152V2 = get_pre_trained_model(
        ResNet152V2, "imagenet", 64, image_size, channels)

    print(
        f"Number of layer ResNet152V2: {len(pre_trained_model_ResNet152V2.layers)}")

    # Configure model
    x = Flatten()(pre_trained_model_ResNet152V2.output)

    # Fully Connection Layers

    # FC1
    x = Dense(1024, activation="relu")(x)

    # FC2
    x = Dense(1024, activation="relu")(x)

    # FC3
    x = Dense(1024, activation="relu")(x)

    # FC4
    x = Dense(1024, activation="relu")(x)

    # #Dropout to avoid overfitting effect
    x = Dropout(0.2)(x)

    # FC5
    x = Dense(512, activation="relu")(x)

    # FC6
    x = Dense(512, activation="relu")(x)

    # FC7
    x = Dense(256, activation="relu")(x)

    # FC8
    x = Dense(256, activation="relu")(x)

    # Dropout to avoid overfitting effect
    x = Dropout(0.2)(x)

    # output layer
    x = Dense(10, activation="softmax")(x)

    # concatenation layers
    model_ResNet152V2 = Model(pre_trained_model_ResNet152V2.input, x)

    # Compile
    model_ResNet152V2.compile(optimizer="adam",
                              loss="categorical_crossentropy",
                              metrics=['accuracy'])

    return model_ResNet152V2


def adj_model_VGG16(image_size, channels):
    # Get pre trained VGG16
    pre_trained_model_VGG16 = get_pre_trained_model(
        VGG16, None, 5, image_size, channels)

    print(f"Number of layer VGG16: {len(pre_trained_model_VGG16.layers)}")

    # Configure
    x = Flatten()(pre_trained_model_VGG16.output)

    # Fully Connection Layer

    # FC1
    x = Dense(1024, activation="relu")(x)

    # FC2
    x = Dense(1024, activation="relu")(x)

    # FC3
    x = Dense(1024, activation="relu")(x)

    # Dropout to avoid overfitting effect
    x = Dropout(0.5)(x)

    # FC4
    x = Dense(512, activation="relu")(x)

    # FC5
    x = Dense(512, activation="relu")(x)

    # Dropout to avoid overfitting effect
    x = Dropout(0.4)(x)

    # FC6
    x = Dense(256, activation="relu")(x)

    # FC7
    x = Dense(64, activation="relu")(x)

    # FC8
    x = Dense(64, activation="relu")(x)

    # Dropout to avoid overfitting effect
    x = Dropout(0.2)(x)

    # output layer
    x = Dense(10, activation="softmax")(x)

    # concatenation layers
    model_VGG16 = Model(pre_trained_model_VGG16.input, x)

    # RMSPorp Optimization
    opt_rms_prop = tf.keras.optimizers.RMSprop(
        learning_rate=0.0001,
        momentum=0.0001,
        epsilon=1e-07,
        name="RMSprop",
    )

    # Compile
    model_VGG16.compile(optimizer=opt_rms_prop,
                        loss="categorical_crossentropy",
                        metrics=['accuracy'])

    return model_VGG16


def evaluate_model(model, dataX, datay, epochs, batch_size, n_folds=5):
    '''
    Divide o dataset usando K-Fold igual 5;
    Faz o fit do modelo e avalia o desempenho
    '''
    scores, histories = [], []
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, test_idx in kfold.split(dataX):
        #model = define_model()
        trainX, trainy, testX, testy = dataX[train_idx], datay[train_idx], dataX[test_idx], datay[test_idx]
        history = model.fit(x=trainX,
                            y=trainy,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(testX, testy),
                            verbose=1
                            )
        _, acc = model.evaluate(testX, testy, verbose=0)
        print(f'Accuracy: {(acc * 100.0):.2f} %')
        scores.append(acc)
        histories.append(history)
    return scores, histories


def summarize_diagnostics(histories):
    for i in range(len(histories)):
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'],
                 color='orange', label='test')

        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'],
                 color='orange', label='test')

    plt.show()


def summarize_performance(scores):
    print(
        f'Accuracy: mean={(np.mean(scores)*100):.2f}% std={(np.std(scores)*100):.2f}% n={len(scores)}')
    plt.boxplot(scores)
    plt.show()


def create_trace(x, y, ylabel, color):
    trace = go.Scatter(
        x=x, y=y,
        name=ylabel,
        marker=dict(color=color),
        mode="markers+lines",
        text=x
    )
    return trace


def plot_acc_loss(history_model, model_name, epochs):
    print(f"- Accuracy and loss for {model_name} model with {epochs} epochs")

    hist = history_model.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1, len(acc) + 1))

    trace_ta = create_trace(epochs, acc, "Training accuracy", "Green")
    trace_va = create_trace(epochs, val_acc, "Validation accuracy", "Red")
    trace_tl = create_trace(epochs, loss, "Training loss", "Blue")
    trace_vl = create_trace(epochs, val_loss, "Validation loss", "Magenta")

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Training and validation accuracy',
                                                        'Training and validation loss'))
    fig.append_trace(trace_ta, 1, 1)
    fig.append_trace(trace_va, 1, 1)
    fig.append_trace(trace_tl, 1, 2)
    fig.append_trace(trace_vl, 1, 2)
    fig['layout']['xaxis'].update(title='Epoch')
    fig['layout']['xaxis2'].update(title='Epoch')
    fig['layout']['yaxis'].update(title='Accuracy')  # , range=[0, 1])
    fig['layout']['yaxis2'].update(title='Loss')  # , range=[0, 1])

    iplot(fig, filename=f'history-{model_name}-accuracy-loss')


def show_predict(pred_model, dataset_X, dataset_Y, model_name):
    # Process on Prediction values for Model
    pred_values = np.argmax(pred_model, axis=1)
    print(f"Predições para o modelo {model_name}: ", pred_values)
    # Display pictures
    plt.figure(figsize=(30, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.title(
            f"Predições: {CLASS_NAMES[pred_values[i]]} <==> Verdade: {CLASS_NAMES[dataset_Y[i]]}")
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(dataset_X[i])
    plt.show()


def show_classification_report(test_Y, pred_Y, class_names, model_name):
    cr = classification_report(
        test_Y, np.argmax(pred_Y, axis=1), target_names=class_names)
    print(
        f'Relatório de classificação para modelo {model_name}: \n ', cr)
    return classification_report(
        test_Y, np.argmax(pred_Y, axis=1), output_dict=True)


def show_classification_report_svm(test_Y, pred_Y, class_names, model_name):
    cr = classification_report(
        test_Y, pred_Y, target_names=class_names)
    print(
        f'Relatório de classificação para modelo {model_name}: \n ', cr)
    return classification_report(
        test_Y, pred_Y, output_dict=True)


def add_model_metrics(cr, train_duration, predict_duration, model_name, idx):
    df = pd.DataFrame()
    metrics = dict(
        model=model_name,
        model_accuracy=cr['accuracy'],
        precision_macro_avg=cr['macro avg']['precision'],
        recall_macro_avg=cr['macro avg']['recall'],
        f1_score_macro_avg=cr['macro avg']['f1-score'],
        support_macro_avg=cr['macro avg']['support'],
        precision_weighted_avg=cr['weighted avg']['precision'],
        recall_weighted_avg=cr['weighted avg']['recall'],
        f1_score_weighted_avg=cr['weighted avg']['f1-score'],
        support_weighted_avg=cr['weighted avg']['support'],
        train_duration = train_duration,
        predict_duration = predict_duration
    )
    df = pd.DataFrame(metrics, index=[idx])
    return df