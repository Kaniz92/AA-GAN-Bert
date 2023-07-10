import argparse

import torch

from src.ganbert.gan_bert_model import GANBertModel


def main():
    default_model_params = {
        'optimizer': 'adam',
        'num_hidden_layers_g': 1,
        'num_hidden_layers_d': 1,
        'out_dropout_rate': 0.2,
        'batch_size': 8,
        'num_train_epochs': 5,
        'warmup_proportion': 0.1,
        'learning_rate_discriminator': 1e-5,
        'learning_rate_generator': 1e-5,
    }

    model_class_selection = {
        'gan_bert': GANBertModel
    }

    parser = argparse.ArgumentParser(description='model params')
    parser.add_argument('--wandb_project_name', required=False, help='wandb_project_name', default="Test_28_11_2022")
    parser.add_argument('--model_class', required=False, help='model_class', default="gan_bert")
    parser.add_argument('--model_type', required=False, help='model_type', default="bert")
    parser.add_argument('--model_name', required=False, help='model_name', default="bert-base-cased")
    parser.add_argument('--sweep_config_enable', required=False, help='sweep_config_enable', default=True)
    parser.add_argument('--sampling_strategy', required=False, help='sampling_strategy',
                        default="single_window_middle_sample")
    parser.add_argument('--training_strategy', required=False, help='training_strategy', default="wandb")
    parser.add_argument('--model_params', required=False, help='model_params', default=default_model_params)
    parser.add_argument('--dataset', required=False, help='dataset', default='experiment_1_2_authors_Trial_5')
    parser.add_argument('--sample_size', required=False, help='sample_size', default=50)
    parser.add_argument('--dataset_dir', required=False, help='dataset_dir', default='authors_2/trial_1')
    parser.add_argument('--output_dir', required=False, help='output_dir', default=None)
    parser.add_argument('--evaluate', required=False, help='evaluate', default=False)
    parser.add_argument('--num_train_epochs', required=False, help='num_train_epochs', default=None)

    # args for random sampling
    # reuse dataset_dir to set masterdata path from huggingface
    parser.add_argument('--random_sampling_indexes_file', required=False, help='random_sampling_indexes_file',
                        default='data/experiment_2/random_sampling_10_split_index.csv')
    parser.add_argument('--case_id', required=False, help='case_1', default=2)
    parser.add_argument('--random_sample_size', required=False, help='random sample size', default=50)

    args = parser.parse_args()

    wandb_project_name = args.wandb_project_name
    model_type = args.model_type
    model_name = args.model_name
    sweep_config_enable = args.sweep_config_enable
    sampling_strategy = args.sampling_strategy
    sample_size = args.sample_size
    training_strategy = args.training_strategy
    model_params = args.model_params
    model_class = model_class_selection[args.model_class]
    dataset = args.dataset
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    evaluate = args.evaluate  # TODO: remove if not required. Now train and testing is combined?

    # args for random sampling
    random_sampling_indexes_file = args.random_sampling_indexes_file
    case_id = args.case_id
    random_sample_size = int(args.random_sample_size)

    if args.num_train_epochs is not None:
        model_params['num_train_epochs'] = int(args.num_train_epochs)

    # TODO: remove or refactor
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # TODO: Pass the model name in body
    # model_name = "bert-base-cased"
    # model_name = "bert-base-uncased"
    # model_name = "roberta-base"
    # model_name = "albert-base-v2"
    # model_name = "xlm-roberta-base"
    # model_name = "amazon/bort"

    model = model_class(
        model_type=model_type,
        model_name=model_name,
        dataset=dataset,
        dataset_dir=dataset_dir,
        sampling_strategy=sampling_strategy,
        sample_size=sample_size,
        use_cuda=torch.cuda.is_available(),
        sweep_config_enable=sweep_config_enable,
        wandb_project_name=wandb_project_name,
        output_dir=output_dir,
        evaluate=evaluate,
        random_sampling_indexes_file=random_sampling_indexes_file,
        training_strategy=training_strategy,
        random_sample_size=random_sample_size
    )

    if training_strategy == 'wandb':
        model.train_with_sweep(output_dir)
    elif training_strategy == 'best_modal_on_wandb':
        model.best_model_train(model_params, output_dir, is_wandb=True)
    elif training_strategy == 'best_modal':
        model.best_model_train(model_params, output_dir)
    elif training_strategy == 'random_sampling_wandb':
        model.random_sampling_with_wandb(output_dir, int(case_id))
    elif training_strategy == 'zero_shots_testing':
        model.zero_shots_testing(model_params)
    elif training_strategy == 'zero_shots_testing_random_samples':
        model.zero_shots_testing_with_wandb_sweep(output_dir, int(case_id))


if __name__ == "__main__":
    main()
