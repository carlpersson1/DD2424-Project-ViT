import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import get_cifar_dataset_scaled, VIT_dataprepocessing_model_phase, ViT_Hierarchical_Architecture
import optuna

def getdata(batch_size=32):
    # main flow
    data = get_cifar_dataset_scaled(batch_size=batch_size)

    training_data = data['training_data']
    validation_data = data['validation_data']
    testing_data = data['testing_data']

    training_data_inputs = np.array(training_data['images'])
    training_data_outputs = np.array(training_data['labels'])

    validation_data_inputs = np.array(validation_data['images'])
    validation_data_outputs = np.array(validation_data['labels'])

    testing_data_inputs = np.array(testing_data['images'])
    testing_data_outputs = np.array(testing_data['labels'])

    onehot_matrix = np.zeros((*training_data_inputs.shape[:2], 10))

    for idx, sample in enumerate(training_data_outputs):
        for idx2, batchnr in enumerate(sample):
            onehot_matrix[idx, idx2, batchnr] = 1

    training_data_outputs = onehot_matrix

    onehot_matrix_validation = np.zeros((*validation_data_outputs.shape[:2], 10))

    for idx, sample in enumerate(validation_data_outputs):
        for idx2, batchnr in enumerate(sample):
            onehot_matrix_validation[idx, idx2, batchnr] = 1

    validation_data_outputs = onehot_matrix_validation

    onehot_matrix_test = np.zeros((*testing_data_outputs.shape[:2], 10))

    for idx, sample in enumerate(testing_data_outputs):
        for idx2, batchnr in enumerate(sample):
            onehot_matrix_test[idx, idx2, batchnr] = 1

    testing_data_outputs = onehot_matrix_test

    return training_data_inputs, training_data_outputs, validation_data_inputs, \
           validation_data_outputs, testing_data_inputs, testing_data_outputs
def run():

    baseline_hyperparams = {'n_channels_encoder_block': 2,
                            # 'hidden_nodes_encoder_block': 40,
                            'dropout_encoder_block': 0.1,
                            'epochs':50,
                            'lr': 0.001,
                            'patch_qty': 16,
                            'n_encoder_blocks': 12,
                            'dimension_dense_projection': 240,
                            'batch_size': 32}

    results_per_trial = {}

    def objective(trial):
        n_channels_encoder_block = trial.suggest_categorical('n_channels_encoder_block', [2, 4, 8])
        # hidden_nodes_encoder_block = trial.suggest_int('hidden_nodes_encoder_block', 2, 128)
        dropout_encoder_block = trial.suggest_float('dropout_encoder_block', 0.01, 0.3)
        patch_qty = trial.suggest_categorical('patch_qty', [4, 16, 64])
        n_encoder_blocks = trial.suggest_int('n_encoder_blocks', 2, 12)

        dimension_dense_projection = trial.suggest_categorical('dimension_dense_projection', [16, 32, 64, 128, 240, 512])

        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = trial.suggest_int('epochs', 1, 50)
        lr = trial.suggest_float('lr', 0.0001, 0.05)
        import math
        size_per_patch = int(math.sqrt(3072 / (3 * patch_qty)))

        training_data_inputs, training_data_outputs, validation_data_inputs, \
        validation_data_outputs, testing_data_inputs, testing_data_outputs = getdata(batch_size=batch_size)

        import tensorflow as tf

        if tf.__version__ >= "2":
            monitor = "val_accuracy"
        else:
            monitor = "val_acc"

        config = {'x_train': training_data_inputs,
                  'y_train': training_data_outputs,
                  'x_val': validation_data_inputs,
                  'y_val': validation_data_outputs,
                  'x_test': testing_data_inputs,
                  'y_test': testing_data_outputs,
                  'n_channels_encoder_block': n_channels_encoder_block,
                  'dropout_encoder_block': dropout_encoder_block,
                  'patch_qty': patch_qty,
                  'n_encoder_blocks': n_encoder_blocks,
                  'dimension_dense_projection': dimension_dense_projection,
                  'size_per_patch': size_per_patch,
                  'epochs': epochs,
                  'lr': lr,
                  'trial': trial,
                  'monitor': monitor}

        results = VIT_dataprepocessing_model_phase(**config)
        
        results_per_trial[trial.number] = results[1:]

        return results[1][0]

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=10))

    study.enqueue_trial(baseline_hyperparams)

    study.optimize(objective, n_trials=100)

    print(results_per_trial)

    final_results_best_model = VIT_dataprepocessing_model_phase(**study.best_params)

    # print(data_preprocessed_model.summary())


def test_run():
    if __name__ == '__main__':
        training_data_inputs, training_data_outputs, validation_data_inputs, \
            validation_data_outputs, testing_data_inputs, testing_data_outputs = getdata(batch_size=128)

        ViTconfig = {'x_train': training_data_inputs,
                  'y_train': training_data_outputs,
                  'x_val': validation_data_inputs,
                  'y_val': validation_data_outputs,
                  'x_test': testing_data_inputs,
                  'y_test': testing_data_outputs,
                  'n_channels_encoder_block': 8,
                  'dropout_encoder_block': 0.1,
                  'L2_reg_encoder_block': 0.0006,
                  'patch_qty': 64,
                  'n_encoder_blocks': 6,
                  'dimension_dense_projection': 128,
                  'size_per_patch': 4,
                  'epochs': 100,
                  'lr': 0.001}

        Hybridconfig = {'x_train': training_data_inputs,
                     'y_train': training_data_outputs,
                     'x_val': validation_data_inputs,
                     'y_val': validation_data_outputs,
                     'x_test': testing_data_inputs,
                     'y_test': testing_data_outputs,
                     'n_channels_encoder_block': 8,
                     'dropout_encoder_block': 0.1,
                     'L2_reg_encoder_block': 0.00,
                     'patch_qty': 256,
                     'n_encoder_blocks': 6,
                     'dimension_dense_projection': 48,
                     'size_per_patch': 2,
                     'epochs': 100,
                     'lr': 0.001}

        results = VIT_dataprepocessing_model_phase(**ViTconfig)
        results = ViT_Hierarchical_Architecture(**Hybridconfig)


def plot():
    # Get model train data
    dict = np.load('ModelStats/VanillaViT.npy', allow_pickle=True).item()
    van_loss = dict['loss']
    van_acc = dict['accuracy']
    van_val_loss = dict['val_loss']
    van_val_acc = dict['val_accuracy']
    dict = np.load('ModelStats/HierarchicalViT.npy', allow_pickle=True).item()
    hier_loss = dict['loss']
    hier_acc = dict['accuracy']
    hier_val_loss = dict['val_loss']
    hier_val_acc = dict['val_accuracy']

    # Plot
    x = np.arange(0, 100, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x, van_acc, label='Vanilla ViT (train)')
    ax1.plot(x, hier_acc, label='Hierarchical ViT (train)')
    ax1.plot(x, van_val_acc, label='Vanilla ViT (validation)')
    ax1.plot(x, hier_val_acc, label='Hierarchical ViT (validation)')
    ax1.set_title('Accuracy during training')
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.plot(x, van_loss, label='Vanilla ViT (train)')
    ax2.plot(x, hier_loss, label='Hierarchical ViT (train)')
    ax2.plot(x, van_val_loss, label='Vanilla ViT (validation)')
    ax2.plot(x, hier_val_loss, label='Hierarchical ViT (validation)')
    ax2.set_title('Loss during training')
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.show()


if __name__ == '__main__':
    run()


