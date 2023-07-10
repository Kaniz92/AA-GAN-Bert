random_sampling_10_args = {
    'method': 'grid',
    'parameters': {
        'set_id': {
            'min': 1,
            'max': 10
        },
        'optimizer': {
            'value': 'adam'
        },
        'num_hidden_layers_g': {
            'value': 1
        },
        'num_hidden_layers_d': {
            'value': 1
        },
        'out_dropout_rate': {
            'value': 0.2
        },
        'batch_size': {
            'value': 8
        },
        'num_train_epochs': {
            'value': 5
        },
        'warmup_proportion': {
            'value': 0.1
        },
        'learning_rate_discriminator': {
            'value': 1e-5
        },
        'learning_rate_generator': {
            'value': 1e-5
        }
    },
    'metric': {
        'name': 'testing_accuracy',
        'goal': 'maximize'
    }
}

random_sampling_50_args = {
    'method': 'grid',
    'parameters': {
        'set_id': {
            'min': 1,
            'max': 50
        },
        'optimizer': {
            'value': 'adam'
        },
        'num_hidden_layers_g': {
            'value': 1
        },
        'num_hidden_layers_d': {
            'value': 1
        },
        'out_dropout_rate': {
            'value': 0.2
        },
        'batch_size': {
            'value': 8
        },
        'num_train_epochs': {
            'value': 5
        },
        'warmup_proportion': {
            'value': 0.1
        },
        'learning_rate_discriminator': {
            'value': 1e-5
        },
        'learning_rate_generator': {
            'value': 1e-5
        }
    },
    'metric': {
        'name': 'testing_accuracy',
        'goal': 'maximize'
    }
}

random_sampling_100_args = {
    'method': 'grid',
    'parameters': {
        'set_id': {
            'min': 1,
            'max': 100
        },
        'optimizer': {
            'value': 'adam'
        },
        'num_hidden_layers_g': {
            'value': 1
        },
        'num_hidden_layers_d': {
            'value': 1
        },
        'out_dropout_rate': {
            'value': 0.2
        },
        'batch_size': {
            'value': 8
        },
        'num_train_epochs': {
            'value': 5
        },
        'warmup_proportion': {
            'value': 0.1
        },
        'learning_rate_discriminator': {
            'value': 1e-5
        },
        'learning_rate_generator': {
            'value': 1e-5
        }
    },
    'metric': {
        'name': 'testing_accuracy',
        'goal': 'maximize'
    }
}

random_sampling_150_args = {
    'method': 'grid',
    'parameters': {
        'set_id': {
            'min': 1,
            'max': 150
        },
        'optimizer': {
            'value': 'adam'
        },
        'num_hidden_layers_g': {
            'value': 1
        },
        'num_hidden_layers_d': {
            'value': 1
        },
        'out_dropout_rate': {
            'value': 0.2
        },
        'batch_size': {
            'value': 8
        },
        'num_train_epochs': {
            'value': 5
        },
        'warmup_proportion': {
            'value': 0.1
        },
        'learning_rate_discriminator': {
            'value': 1e-5
        },
        'learning_rate_generator': {
            'value': 1e-5
        }
    },
    'metric': {
        'name': 'testing_accuracy',
        'goal': 'maximize'
    }
}
