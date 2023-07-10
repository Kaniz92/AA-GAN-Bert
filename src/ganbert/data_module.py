import ast

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from src.data_sampling.single_window_sample import middle_sample, n_samples_sequential


class GANDataModule:
    def __init__(
            self,
            dataset_name,
            dataset_dir,
            batch_size,
            apply_balance,
            max_seq_length,
            tokenizer,
            sampling_strategy,
            sample_size,
            random_sampling_indexes_file,
            training_strategy,
            set_id,
            case_id
    ):
        super().__init__()
        self.label_list = None
        self.test_label_masks = None
        self.train_label_masks = None
        self.train_examples = None
        self.label_map = None
        self.dataset = None
        self.val_examples = None
        self.test_examples = None
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.apply_balance = apply_balance
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.dataset_dir = dataset_dir
        self.random_sampling_indexes_file = random_sampling_indexes_file
        self.training_strategy = training_strategy
        self.set_id = set_id
        self.case_id = case_id

        # If random sampling, we create dataset from book ids per each set from wandb config
        # else we go with the normal flow prepare_data()

        if training_strategy == 'random_sampling_wandb' or training_strategy == 'zero_shots_testing_random_samples':
            self.create_dataset()
        else:
            self.prepare_data()
        self.setup()

        sampling_strategy_select = {
            'single_window_middle_sample': middle_sample,
            'n_samples_sequential': n_samples_sequential
        }

        self.sampling_class = sampling_strategy_select[sampling_strategy]

    def create_dataset(self):
        #     load master data
        master_dataset = load_dataset(f"Authorship/{self.dataset_name}", data_files="main_data.csv")

        master_dataset = pd.DataFrame(master_dataset['train'])
        master_dataset = master_dataset[master_dataset['Genre'] == 'Novel']

        # load book indexes
        data_indexes = pd.read_csv(self.random_sampling_indexes_file)

        # 1. get row from case_id and set_id
        data_index_set_row = data_indexes[
            (data_indexes['case_id'] == self.case_id) & (data_indexes['set_id'] == self.set_id)]

        # 2. get train,test,val indexes
        # 3. get dataset splits from master_data
        train_data = master_dataset[master_dataset['BookID'].isin(ast.literal_eval(data_index_set_row.train.values[0]))]
        val_data = master_dataset[master_dataset['BookID'].isin(ast.literal_eval(data_index_set_row.val.values[0]))]
        test_data = master_dataset[master_dataset['BookID'].isin(ast.literal_eval(data_index_set_row.test.values[0]))]

        # 4. create self.dataset
        self.dataset = {'train': train_data, 'test': test_data, 'validation': val_data}

        self.label_list = train_data['AuthorID'].unique().tolist()

        # TODO: get label_list from unique author ids
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i

    def prepare_data(self):
        if self.dataset_name == 'Baselines':
            data  = load_dataset(f"Authorship/{self.dataset_name}", data_files=self.dataset_dir)
            data = pd.DataFrame(data['train'])
            trainval, test = train_test_split(data, 
                                  test_size=0.2,
                                  stratify=data['AuthorID'])    
            train, val = train_test_split(trainval, test_size=0.1, stratify=trainval['AuthorID'])
            self.dataset = {}
            self.dataset['train'] = train
            self.dataset['validation'] = val
            self.dataset['test'] = test

        else:
            self.dataset = load_dataset(f"Authorship/{self.dataset_name}", data_dir=self.dataset_dir)
        train_data = pd.DataFrame(self.dataset['train'])

        self.label_list = train_data['AuthorID'].unique().tolist()

        # TODO: get label_list from unique author ids
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i

    def get_label_list(self):
        return self.label_list

    def generate_data_loader(self, input_examples, label_map, do_shuffle=False):
        input_ids = []
        input_mask_array = []
        label_mask_array = []
        label_id_array = []

        # Tokenization
        for (idx, text) in input_examples.iterrows():
            sample_text_list = self.sampling_class(text['BookText'], self.max_seq_length, self.sample_size)

            for sample_text in sample_text_list:
                if sample_text is not None:
                    encoded_sent = self.tokenizer.encode(sample_text, add_special_tokens=True,
                                                         max_length=self.max_seq_length, padding="max_length",
                                                         truncation=True)
                    del sample_text

                    input_ids.append(encoded_sent)
                    del encoded_sent

                    label = label_map[text['AuthorID']]
                    label_id_array.append(label)
                    label_mask_array.append(True)  # 1 for all labeled data

        # Attention to token (to ignore padded input workpieces)
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            input_mask_array.append(att_mask)
        # Conversion to Tensor
        input_ids = torch.tensor(input_ids)
        input_mask_array = torch.tensor(input_mask_array)
        label_id_array = torch.tensor(label_id_array, dtype=torch.long)

        label_mask_array = torch.tensor(label_mask_array)

        # Building the TensorDataset
        dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)
        del input_ids, input_mask_array, label_id_array, label_mask_array

        if do_shuffle:
            sampler = RandomSampler
        else:
            sampler = SequentialSampler

        # Building the DataLoader
        return DataLoader(
            dataset,  # The training samples.
            sampler=sampler(dataset),
            batch_size=self.batch_size)

    def setup(self):
        unlabeled_examples = []

        self.train_examples = pd.DataFrame(self.dataset['train'])
        self.train_label_masks = np.ones(len(self.train_examples), dtype=bool)

        self.val_examples = pd.DataFrame(self.dataset['validation'])
        self.test_examples = pd.DataFrame(self.dataset['test'])

        if unlabeled_examples:
            self.train_examples = self.train_examples + unlabeled_examples
            # The unlabeled (train) dataset is assigned with a mask set to False
            tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)
            self.train_label_masks = np.concatenate([self.train_label_masks, tmp_masks])

            self.test_label_masks = np.ones(len(self.val_examples), dtype=bool)

    def train_dataloader(self):
        return self.generate_data_loader(self.train_examples, self.label_map, do_shuffle=True)

    def val_dataloader(self):
        return self.generate_data_loader(self.val_examples, self.label_map, do_shuffle=False)

    def test_dataloader(self):
        return self.generate_data_loader(self.test_examples, self.label_map, do_shuffle=False)
