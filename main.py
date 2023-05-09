
def run():
    # main flow
    from data_preprocessing import get_cifar_dataset_scaled, VIT_dataprepocessing_model_phase
    cifar_10_data = get_cifar_dataset_scaled()
    config_cifar_10 = {'image_size': 32,
                       'patch_qty': 16,
                       'size_per_patch': 16,
                       'dimension_dense_projection': 256,
                       'dense_activation': "relu"}
    data_preprocessed_model = VIT_dataprepocessing_model_phase(**config_cifar_10)
    print(data_preprocessed_model.summary())


if __name__ == '__main__':
    run()

