import logging
import os
import keras
import keras_tuner as kt
import numpy as np
import csv
from datetime import datetime
from keras import layers
from keras.src.optimizers.adam import Adam
from keras.src.optimizers.sgd import SGD
from sklearn.model_selection import StratifiedShuffleSplit

script_dir = os.path.dirname(os.path.abspath(__file__))

logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

log_filename = f"tuning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(logs_dir, log_filename)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('\n%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    print(f"Log file created at: {log_filepath}")
except Exception as e:
    print(f"Failed to create log file. Error: {e}")
    print("Falling back to console logging.")

class AccuracyLogger:
    def __init__(self, filename):
        self.filename = filename
        self.fieldnames = ['timestamp', 'fold', 'inner_fold', 'trial_id', 'accuracy', 'accuracy_type']
        
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_accuracy(self, fold, inner_fold, trial_id, accuracy, accuracy_type):
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'fold': fold,
                'inner_fold': inner_fold,
                'trial_id': trial_id,
                'accuracy': accuracy,
                'accuracy_type': accuracy_type
            })

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        n_units = hp.Int('n_units', 32, 256, step=32)
        n_hidden = hp.Int('n_hidden', 1, 10)
        reg_strength = hp.Float('reg_strength', 1e-10, 1e-2, sampling='log')
        optimizer_str = hp.Choice('optimizer_str', ['sgd', 'adam'])
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        activation = hp.Choice('activation', ['selu', 'silu', 'leaky_relu', 'relu'])

        def get_activation(name):
            if name == 'selu':
                return 'selu'
            elif name == 'silu':
                return 'swish'
            elif name == 'leaky_relu':
                return layers.LeakyReLU(alpha=0.01)
            else:
                return 'relu'

        act_func = get_activation(activation)

        model = keras.Sequential()
        
        model.add(layers.Flatten(input_shape=(28, 28)))

        if activation == 'selu': 
            model.add(layers.Dense(units=n_units,
                                activation=act_func,
                                kernel_initializer='lecun_normal',
                                kernel_regularizer=keras.regularizers.l2(reg_strength)))
        else:
            model.add(layers.Dense(units=n_units,
                                activation=act_func,
                                kernel_regularizer=keras.regularizers.l2(reg_strength)))
        model.add(layers.BatchNormalization())

        for _ in range(n_hidden - 1): 
            if activation == 'selu': 
                model.add(layers.Dense(units=n_units,
                                    activation=act_func,
                                    kernel_initializer='lecun_normal',
                                    kernel_regularizer=keras.regularizers.l2(reg_strength)))
            else:
                model.add(layers.Dense(units=n_units,
                                    activation=act_func,
                                    kernel_regularizer=keras.regularizers.l2(reg_strength)))
            model.add(layers.BatchNormalization())
        
        model.add(layers.Dense(10, activation='softmax'))

        if optimizer_str == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, clipnorm=1.0)
        else:
            optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
    
class LoggingRandomSearch(kt.RandomSearch):
    def __init__(self, accuracy_logger, fold, inner_fold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_logger = accuracy_logger
        self.fold = fold
        self.inner_fold = inner_fold

    def on_trial_end(self, trial):
        val_accuracy = trial.metrics.get_last_value('val_accuracy')
        self.accuracy_logger.log_accuracy(
            self.fold, 
            self.inner_fold, 
            trial.trial_id, 
            val_accuracy, 
            'validation'
        )
        super().on_trial_end(trial)

def run_experiment(x, y, n_splits=3, n_inner_splits=3, max_epochs=50, max_trials=100, seed=42):
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=seed)
    logger.info(f"Starting experiment with {n_splits} outer splits and {n_inner_splits} inner splits")
    accuracy_logger = AccuracyLogger(f"accuracy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    start_time = datetime.now()

    all_fold_results = []

    for fold, (train_index, test_index) in enumerate(outer_cv.split(x,y), 1):
        fold_start_time = datetime.now()
        logger.info(f"Starting fold {fold}")

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_cv = StratifiedShuffleSplit(n_splits=n_inner_splits, test_size=0.3, random_state=seed)

        inner_fold_results = []
        
        for inner_fold, (inner_train_index, inner_val_index) in enumerate(inner_cv.split(x_train, y_train), 1):
            x_inner_train, x_val = x_train[inner_train_index], x_train[inner_val_index]
            y_inner_train, y_val = y_train[inner_train_index], y_train[inner_val_index]

            tuner_dir = f'random_search_tuning_fold_{fold}_inner_{inner_fold}'
            project_name = f'fashion_mnist_random_search_fold_{fold}_{inner_fold}'

            hypermodel = MyHyperModel()
            inner_tuner = LoggingRandomSearch(
                hypermodel=hypermodel,
                objective='val_accuracy',
                max_trials=max_trials,
                directory=tuner_dir,
                project_name=project_name,
                overwrite=True,
                accuracy_logger=accuracy_logger,
                fold=fold,
                inner_fold=inner_fold,
                seed=seed
            )
        
            inner_start_time = datetime.now()
            inner_tuner.search(x_inner_train, y_inner_train, validation_data=(x_val, y_val), epochs=20)  
            inner_end_time = datetime.now()
            inner_time = inner_end_time - inner_start_time

            best_hp = inner_tuner.get_best_hyperparameters()[0]
            best_model = inner_tuner.hypermodel.build(best_hp)
            best_model.fit(x_inner_train, y_inner_train, epochs=max_epochs, validation_data=(x_val, y_val))    

            _, val_accuracy = best_model.evaluate(x_val, y_val)
            logger.info(f"Fold {fold}.{inner_fold} completed in {inner_time} with validation accuracy: {val_accuracy} and config: {best_hp.values}")

            inner_fold_results.append({
                'inner_fold': inner_fold,
                'best_hp': best_hp,
                'val_accuracy': val_accuracy
            })

            accuracy_logger.log_accuracy(fold, inner_fold, 'best', val_accuracy, 'validation')

        best_inner_result = max(inner_fold_results, key=lambda x: x['val_accuracy'])
        best_hp = best_inner_result['best_hp']

        logger.info(f"Best hyperparameters for fold {fold}: {best_hp.values}")

        best_model = hypermodel.build(best_hp)
        best_model.fit(x_train, y_train, epochs=max_epochs)

        _, test_accuracy = best_model.evaluate(x_test, y_test)
        accuracy_logger.log_accuracy(fold, 0, 'final', test_accuracy, 'test')

        fold_end_time = datetime.now()
        fold_time = fold_end_time - fold_start_time
        logger.info(f"Fold {fold} completed in {fold_time} with test accuracy: {test_accuracy}")

        all_fold_results.append({
            'fold': fold,
            'best_hp': best_hp,
            'test_accuracy': test_accuracy,
            'fold_time': fold_time
        })

    end_time = datetime.now()
    total_time = end_time - start_time
    logger.info(f"Experiment completed in {total_time}")

    mean_accuracy = np.mean([result['test_accuracy'] for result in all_fold_results])
    std_accuracy = np.std([result['test_accuracy'] for result in all_fold_results])
    logger.info(f"Mean test accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

    return all_fold_results

(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data() 
x = np.concatenate([x_train, x_val])
y = np.concatenate([y_train, y_val])
x = x.astype('float32') / 255.0

run_experiment(x=x, y=y, n_splits=3, n_inner_splits=3, max_epochs=50, max_trials=100, seed=8)  