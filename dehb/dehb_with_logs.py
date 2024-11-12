import logging
import shutil
import os
import keras
import keras_tuner as kt
import numpy as np
import csv
from datetime import datetime
from keras import layers
from keras.src.optimizers.adam import Adam
from keras.src.optimizers.sgd import SGD
from syne_tune.optimizer.baselines import DEHB
from syne_tune.config_space import choice, uniform, randint
from syne_tune.backend.trial_status import Trial
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

class DEHBHyperModel(kt.HyperModel):
    def build(self, hp: kt.HyperParameters):
        n_units = hp.Int('n_units', 2, 256)
        n_hidden = hp.Int('n_hidden', 1, 50)
        reg_strength = hp.Float('reg_strength', 1e-10, 1e-1)
        optimizer_str = hp.Choice('optimizer_str', ['sgd', 'adam'])
        learning_rate = hp.Float('learning_rate', 1e-5, 1e-1)
        activation = hp.Choice('activation', ['selu', 'silu', 'leaky_relu', 'relu'])
        # conf = { 'activation': activation, 'learning_rate': learning_rate, 'n_hidden': n_hidden, 'n_units': n_units, 'optimizer_str': optimizer_str, 'reg_strength': reg_strength }

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

class DEHBOracle(kt.Oracle):
    def __init__(self, objective, searcher: DEHB, max_trials):
        super().__init__(objective=objective, max_trials=max_trials)
        self.seacher = searcher
        self.suggestions = []
        self.syne_trials = {}

    def populate_space(self, trial_id):
        config = self.seacher.suggest(trial_id).config
        self.suggestions.append(config)
        status = 'RUNNING'
        if config == 'FINISHED':
            status = 'STOPPED'
        return { 'values': config, 'status': status }


class DEHBTuner(kt.Tuner):
    def __init__(self, hypermodel, config_space, objective, max_epochs, max_trials, seed, accuracy_logger, fold, inner_fold, directory=None, project_name=None, **kwargs):
        if directory is None:
            directory = 'dehb_tuning'
        if project_name is None:
            project_name = 'mnist_dehb'
        
        full_path = os.path.join(directory, project_name)
        os.makedirs(full_path, exist_ok=True)

        self.dehb = DEHB(
            config_space=config_space,
            metric=objective.name,
            resource_attr='epoch',
            max_resource_level=max_epochs,
            support_pause_resume=False,
            random_seed=seed)
        self.oracle = DEHBOracle(objective, searcher=self.dehb, max_trials=max_trials)
        self.max_epochs = max_epochs
        self.curr_trial = None
        self.accuracy_logger = accuracy_logger
        self.fold = fold
        self.inner_fold = inner_fold
        self.best_accuracy = 0
        super().__init__(oracle=self.oracle, hypermodel=hypermodel, **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        self.curr_trial = Trial(
            trial_id=trial.trial_id, 
            config=trial.hyperparameters.get_config()['values'], 
            creation_time=datetime.now())
        logger.info(f"Starting trial {trial.trial_id} with config: {self.curr_trial.config}")
        kwargs['epochs'] = self.max_epochs
        return super().run_trial(trial, *args, **kwargs)

    def on_epoch_end(self, trial, model, epoch, logs=None):
        result = { 'epoch': epoch + 1, 'val_accuracy': logs['val_accuracy'] }
        logger.info(f"Trial {trial.trial_id} - Epoch {epoch + 1}: val_accuracy = {logs['val_accuracy']}")
        decision = self.dehb.on_trial_result(trial=self.curr_trial, result=result)

        if decision == 'STOP':
            logger.info(f"Trial {trial.trial_id} stopped early at epoch {epoch + 1}")
            trial.status = 'STOPPED' 
            model.stop_training = True
            self.oracle.end_trial(trial)
            self.dehb.on_trial_complete(self.curr_trial, result=result)

    def on_trial_end(self, trial):
        if trial.status == "COMPLETED":
            val_accuracy = trial.metrics.get_last_value('val_accuracy')
            self.accuracy_logger.log_accuracy(
                self.fold, self.inner_fold, trial.trial_id, val_accuracy, 'validation'
            )
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                logger.info(f"New best config: {trial.hyperparameters.values}")
        return super().on_trial_end(trial)
    

def run_experiment(x, y, n_splits=3, n_inner_splits=3, max_epochs=50, max_trials=100, seed=42):
    config_space = {
        'n_units': randint(32, 256),
        'n_hidden': randint(1, 10),
        'reg_strength': uniform(1e-10, 1e-2),
        'optimizer_str': choice(['sgd', 'adam']),
        'learning_rate': uniform(1e-4, 1e-2),
        'activation': choice(['selu', 'silu', 'leaky_relu', 'relu'])
    }
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

            tuner_dir = f'dehb_tuning_fold_{fold}_inner_{inner_fold}'
            project_name=f'mnist_dehb_fold_{fold}_{inner_fold}'

            hypermodel = DEHBHyperModel()
            inner_tuner = DEHBTuner(
                hypermodel=hypermodel,
                config_space=config_space,
                objective=kt.Objective('val_accuracy', 'max'),
                max_epochs=20,
                max_trials=max_trials,
                seed=seed,
                accuracy_logger=accuracy_logger,
                fold=fold,
                inner_fold=inner_fold,
                directory=tuner_dir,
                project_name=project_name,
                overwrite = True
            )
        
            inner_start_time = datetime.now()
            inner_tuner.search(x_inner_train, y_inner_train, validation_data=(x_val, y_val))  
            inner_end_time = datetime.now()
            inner_time = inner_start_time - inner_end_time

            best_hp = inner_tuner.get_best_hyperparameters()[0]
            best_model = inner_tuner.hypermodel.build(best_hp)
            best_model.fit(x_inner_train, y_inner_train, epochs=20, validation_data=(x_val, y_val))    

            _, val_accuracy = best_model.evaluate(x_val, y_val)
            logger.info(f"Fold {fold}.{inner_fold} completed in {inner_time} with test accuracy: {val_accuracy} and config: {best_hp.get_config()['values']}")

            inner_fold_results.append({
                'inner_fold': inner_fold,
                'best_hp': best_hp,
                'val_accuracy': val_accuracy
            })

        best_inner_result = max(inner_fold_results, key=lambda x: x['val_accuracy'])
        best_hp = best_inner_result['best_hp']

        logger.info(f"Best hyperparameters for fold {fold}: {best_hp.get_config()['values']}")

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
    total_time = start_time - end_time
    logger.info(f"Experiment completed in {total_time}")

    mean_accuracy = np.mean([result['test_accuracy'] for result in all_fold_results])
    std_accuracy = np.std([result['test_accuracy'] for result in all_fold_results])
    logger.info(f"Mean test accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

    return all_fold_results


(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data() 
x = np.concatenate([x_train, x_val])
y = np.concatenate([y_train, y_val])

run_experiment(x=x, y=y, n_splits=3, n_inner_splits=3, max_epochs=50, max_trials=100, seed=8)
