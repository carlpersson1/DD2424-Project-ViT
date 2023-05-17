
import numpy as np
def run():
    # main flow
    from data_preprocessing import get_cifar_dataset_scaled, VIT_dataprepocessing_model_phase
    data = get_cifar_dataset_scaled()

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

    classes = training_data_outputs.shape[-1]

    # image = image.reshape((image_size, image_size, 3), order='F')
    # config_cifar_10 = {'image_size': 32,
    #                    'patch_qty': 32,
    #                    'size_per_patch': 16,
    #                    'dimension_dense_projection': 256,
    #                    'dense_activation': "relu"}
    #
    # x_train,
    # y_train,
    # x_test,
    # y_test,
    config_cifar_10 = {'x_train': training_data_inputs,
                       'y_train': training_data_outputs,
                       'x_test': validation_data_inputs,
                       'y_test': validation_data_outputs}
    data_preprocessed_model = VIT_dataprepocessing_model_phase(**config_cifar_10)

    print(data_preprocessed_model.summary())


if __name__ == '__main__':
    run()

