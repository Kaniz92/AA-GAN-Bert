sweep_config_args = {
    'method': 'grid',
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'num_hidden_layers_g': {
            'values': [1, 2, 3, 4, 5]
        },
        'num_hidden_layers_d': {
            'values': [1, 2, 3, 4, 5]
        },
        'out_dropout_rate': {
            'values': [0.2, 0.3, 0.4, 0.5]
        },
        'batch_size': {
            'values': [8, 16, 32, 64, 128]
        },
        'num_train_epochs': {
            'values': [5, 10, 30, 40, 50, 60, 70, 80, 90, 100]
        },
        'warmup_proportion': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'learning_rate_discriminator': {
            'values': [1e-5, 2e-5, 5e-5]
        },
        'learning_rate_generator': {
            'values': [1e-5, 2e-5, 5e-5]
        }
    },
    'metric': {
        'name': 'testing_accuracy',
        'goal': 'maximize'
    }
}
