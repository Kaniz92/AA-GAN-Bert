import gc
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_constant_schedule_with_warmup
)

from src.config.model_args import GANBertModelArgs
from src.config.random_sampling_sweep_args import *
from src.config.sweep_config_args import sweep_config_args
from src.ganbert.data_module import GANDataModule
from src.ganbert.discriminator import Discriminator
from src.ganbert.generator import Generator
from src.ganbert.utils import format_time

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


class GANBertModel:
    def __init__(
            self,
            model_type,
            model_name,
            sampling_strategy,
            sample_size,
            dataset,
            dataset_dir,
            tokenizer_type=None,
            tokenizer_name=None,
            use_cuda=False,
            sweep_config_enable=False,
            wandb_project_name=None,
            output_dir=None,
            evaluate=False,
            random_sampling_indexes_file='data/experiment_2/random_sampling_10_split_index.csv',
            training_strategy='best_modal_on_wandb',
            random_sample_size=50,
            **kwargs
    ):
        self.val_dataloader = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.label_list = None
        self.case_id = None

        self.training_strategy = training_strategy

        MODEL_CLASSES = {
            # TODO: update model type to auto?
            'bert': (AutoConfig, AutoModel, AutoTokenizer)
        }

        RANDOM_SAMPLE_CONFIGS = {
            10: random_sampling_10_args,
            50: random_sampling_50_args,
            100: random_sampling_100_args,
            150: random_sampling_150_args
        }

        self.random_sampling_args = RANDOM_SAMPLE_CONFIGS[random_sample_size]
        self.args = self._load_model_args(model_name)

        if sweep_config_enable:
            self.is_sweeping = True

            if self.training_strategy == 'random_sampling_wandb' or self.training_strategy == 'zero_shots_testing_random_samples':
                # sweep_config_args = random_sampling_args  # TODO: refactor later
                self.sweep_id = wandb.sweep(self.random_sampling_args, project=wandb_project_name)
            else:
                self.sweep_id = wandb.sweep(sweep_config_args, project=wandb_project_name)

        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.args.manual_seed)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        self.config = config_class.from_pretrained(model_name, **kwargs)

        if use_cuda:
            if torch.cuda.is_available():
                # if cuda_device == -1 or cuda_device == 'cuda':
                self.device = torch.device("cuda")
            # else:
            #     self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.model = model_class.from_pretrained(
            model_name, config=self.config, **kwargs
        )

        self.model_default = self.model

        self.results = {}

        if tokenizer_name is None:
            tokenizer_name = model_name

        self.tokenizer = tokenizer_class.from_pretrained(
            model_name
        )

        self.args.model_name = model_name
        self.args.model_type = model_type
        self.args.tokenizer_name = tokenizer_name
        self.args.tokenizer_type = tokenizer_type
        self.sampling_strategy = sampling_strategy
        self.dataset = dataset
        self.sample_size = sample_size
        self.dataset_dir = dataset_dir
        self.output_dir = self.args.output_dir
        self.wandb_project_name = wandb_project_name
        self.evaluate = evaluate
        self.random_sampling_indexes_file = random_sampling_indexes_file

        if output_dir is not None:
            self.output_dir = output_dir

        if wandb_project_name and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.wandb_project = None

    def evaluate_model(self, config):
        set_id = None

        if self.training_strategy == 'random_sampling_wandb' or self.training_strategy == 'zero_shots_testing_random_samples':
            set_id = config['set_id']

        # TODO: pass set id from params
        GAN_data_module = GANDataModule(self.dataset,
                                        self.dataset_dir,
                                        config['batch_size'],
                                        self.args.apply_balance,
                                        self.args.max_seq_length,
                                        self.tokenizer,
                                        self.sampling_strategy,
                                        self.sample_size,
                                        self.random_sampling_indexes_file,
                                        self.training_strategy,
                                        set_id,
                                        self.case_id
                                        )

        self.test_dataloader = GAN_data_module.test_dataloader()
        self.label_list = GAN_data_module.get_label_list()
        num_labels = len(self.label_list)

        # If we don't same the label size as the base model (20-authors case) the discriminator will throw an error for mismatching output sizes
        # Therefore we are setting label size as 20
        # TODO: what if we need to do zero shot testing from smaller author size to a higher author size
        if self.training_strategy == 'zero_shots_testing' or self.training_strategy == 'zero_shots_testing_random_samples':
            num_labels = 20

        # model = AutoModel.from_pretrained(self.args.model_name)
        hidden_size = int(self.config.hidden_size)
        hidden_levels_g = [hidden_size for i in range(0, config['num_hidden_layers_g'])]
        hidden_levels_d = [hidden_size for i in range(0, config['num_hidden_layers_d'])]

        self.generator = Generator(noise_size=self.args.noise_size, output_size=hidden_size,
                                   hidden_sizes=hidden_levels_g,
                                   dropout_rate=config['out_dropout_rate'])
        self.generator.load_state_dict(torch.load(f'{self.args.model_name}/generator.pth'))

        # TODO: in zero shot set num_labels = 20
        self.discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d,
                                           num_labels=num_labels,
                                           dropout_rate=config['out_dropout_rate'])
        self.discriminator.load_state_dict(torch.load(f'{self.args.model_name}/discriminator.pth'))

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.model.cuda()
            if self.args.multi_gpu:
                self.model = torch.nn.DataParallel(self.model)

        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # TODO: load model from best folder or wandb
        # TODO: self GAN model, not transformer model - for now were are saving transformer model
        model = self.model

        print("")
        print('Testing...')

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()  # maybe redundant
        self.discriminator.eval()
        self.generator.eval()

        total_test_loss = 0

        all_preds = []
        all_labels_ids = []

        # loss
        nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Evaluate data for one epoch
        for batch in self.test_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                model_outputs = model(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]
                _, logits, probs = self.discriminator(hidden_states)
                ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
                filtered_logits = logits[:, 0:-1]
                # Accumulate the test loss.
                total_test_loss += nll_loss(filtered_logits, b_labels)

            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        # print(all_preds)
        # print(all_labels_ids)
        # test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
        # print("  Accuracy: {0:.3f}".format(test_accuracy))
        test_accuracy = accuracy_score(all_labels_ids, all_preds)
        print("  Accuracy: {0:.3f}".format(test_accuracy))

        precision = precision_score(all_labels_ids, all_preds, average='macro')
        recall = recall_score(all_labels_ids, all_preds, average='macro')
        f1 = f1_score(all_labels_ids, all_preds, average='macro')
        # precision, recall, f1, support = score(all_labels_ids, all_preds)

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(self.test_dataloader)
        avg_test_loss = avg_test_loss.item()

        # Measure how long the validation run took.
        test_time = format_time(time.time() - t0)

        print("  Test Loss: {0:.3f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))

        if self.is_sweeping:
            wandb.log({
                "testing_loss": avg_test_loss,
                "testing_accuracy": test_accuracy,
                'testing_precision': precision,
                'testing_recall': recall,
                'testing_f1': f1
            })

    def train(self, config):
        set_id = None

        if self.training_strategy == 'random_sampling_wandb':
            set_id = config['set_id']

        GAN_data_module = GANDataModule(self.dataset,
                                        self.dataset_dir,
                                        config['batch_size'],
                                        self.args.apply_balance,
                                        self.args.max_seq_length,
                                        self.tokenizer,
                                        self.sampling_strategy,
                                        self.sample_size,
                                        self.random_sampling_indexes_file,
                                        self.training_strategy,
                                        set_id,
                                        self.case_id
                                        )

        self.train_dataloader = GAN_data_module.train_dataloader()
        self.test_dataloader = GAN_data_module.test_dataloader()
        self.val_dataloader = GAN_data_module.val_dataloader()
        self.label_list = GAN_data_module.get_label_list()

        # model = AutoModel.from_pretrained(self.args.model_name)
        hidden_size = int(self.config.hidden_size)
        hidden_levels_g = [hidden_size for i in range(0, config['num_hidden_layers_g'])]
        hidden_levels_d = [hidden_size for i in range(0, config['num_hidden_layers_d'])]

        self.generator = Generator(noise_size=self.args.noise_size, output_size=hidden_size,
                                   hidden_sizes=hidden_levels_g,
                                   dropout_rate=config['out_dropout_rate'])
        self.discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d,
                                           num_labels=len(self.label_list),
                                           dropout_rate=config['out_dropout_rate'])

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.model.cuda()
            if self.args.multi_gpu:
                self.model = torch.nn.DataParallel(self.model)

        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # models parameters
        transformer_vars = [i for i in self.model.parameters()]
        d_vars = transformer_vars + [v for v in self.discriminator.parameters()]
        g_vars = [v for v in self.generator.parameters()]

        # optimizer
        if config['optimizer'] == "sgd":
            dis_optimizer = torch.optim.SGD(d_vars, lr=config['learning_rate_discriminator'])
            gen_optimizer = torch.optim.SGD(g_vars, lr=config['learning_rate_generator'])
        elif config['optimizer'] == "adam":
            dis_optimizer = torch.optim.AdamW(d_vars, lr=config['learning_rate_discriminator'])
            gen_optimizer = torch.optim.AdamW(g_vars, lr=config['learning_rate_generator'])

        del (transformer_vars, d_vars, g_vars)
        gc.collect()

        num_train_examples = len(self.train_dataloader)

        if self.args.apply_scheduler:
            num_train_steps = int(
                num_train_examples / config['batch_size'] * config['num_train_epochs'])
            num_warmup_steps = int(num_train_steps * config['warmup_proportion'])

            scheduler_d = get_constant_schedule_with_warmup(dis_optimizer,
                                                            num_warmup_steps=num_warmup_steps)
            scheduler_g = get_constant_schedule_with_warmup(gen_optimizer,
                                                            num_warmup_steps=num_warmup_steps)

        offset = random.random() / 5

        for epoch_i in range(0, config['num_train_epochs']):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, config['num_train_epochs']))
            print('Training...')

            t0 = time.time()

            tr_g_loss = 0
            tr_d_loss = 0

            self.model.train()
            self.generator.train()
            self.discriminator.train()

            for step, batch in enumerate(self.train_dataloader):
                if step % self.args.print_each_n_step == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)

                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader),
                                                                                elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)

                # Encode real data in the Transformer
                model_outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]

                # Generate fake data that should have the same distribution of the ones
                # encoded by the transformer.
                # First noisy input are used in input to the Generator
                real_batch_size = b_input_ids.shape[0]
                noise = torch.zeros(real_batch_size, self.args.noise_size, device=self.device).uniform_(0.1, 1)
                # Gnerate Fake data
                gen_rep = self.generator(noise)

                # Generate the output of the Discriminator for real and fake data.
                # First, we put together the output of the tranformer and the generator
                disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
                # Then, we select the output of the disciminator
                features, logits, probs = self.discriminator(disciminator_input)

                # Finally, we separate the discriminator's output for the real and fake
                # data
                features_list = torch.split(features, real_batch_size)
                D_real_features = features_list[0]
                D_fake_features = features_list[1]

                logits_list = torch.split(logits, real_batch_size)
                D_real_logits = logits_list[0]
                D_fake_logits = logits_list[1]

                probs_list = torch.split(probs, real_batch_size)
                D_real_probs = probs_list[0]
                D_fake_probs = probs_list[1]

                # ---------------------------------
                #  LOSS evaluation
                # ---------------------------------
                # Generator's LOSS estimation
                g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:, -1] + self.args.epsilon))
                g_feat_reg = torch.mean(
                    torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
                g_loss = g_loss_d + g_feat_reg

                # Disciminator's LOSS estimation
                logits = D_real_logits[:, 0:-1]
                log_probs = F.log_softmax(logits, dim=-1)
                # The discriminator provides an output for labeled and unlabeled real data
                # so the loss evaluated for unlabeled data is ignored (masked)

                b_labels = batch[2].to(self.device)
                b_label_mask = batch[3].to(self.device)
                label2one_hot = torch.nn.functional.one_hot(b_labels, len(self.label_list))
                per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
                per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(self.device))
                labeled_example_count = per_example_loss.type(torch.float32).numel()

                # It may be the case that a batch does not contain labeled examples,
                # so the "supervised loss" in this case is not evaluated
                if labeled_example_count == 0:
                    D_L_Supervised = 0
                else:
                    D_L_Supervised = torch.div(torch.sum(per_example_loss.to(self.device)),
                                               labeled_example_count)

                D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + self.args.epsilon))
                D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + self.args.epsilon))
                d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U
                # d_loss = D_L_unsupervised1U + D_L_unsupervised2U

                # ---------------------------------
                #  OPTIMIZATION
                # ---------------------------------
                # Avoid gradient accumulation
                gen_optimizer.zero_grad()
                dis_optimizer.zero_grad()

                # Calculate weigth updates
                # retain_graph=True is required since the underlying graph will be deleted after backward
                g_loss.backward(retain_graph=True)
                d_loss.backward()

                # Apply modifications
                gen_optimizer.step()
                dis_optimizer.step()

                # A detail log of the individual losses
                # print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".
                #      format(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U,
                #             g_loss_d, g_feat_reg))

                # Save the losses to print them later
                tr_g_loss += g_loss.item()
                tr_d_loss += d_loss.item()

                # Update the learning rate with the scheduler
                if self.args.apply_scheduler:
                    scheduler_d.step()
                    scheduler_g.step()

                # save global step checkpoint
                # if step % self.args.print_each_n_step == 0 and not step == 0:
                #   # Save model checkpoint
                #   output_dir_current = os.path.join(
                #       self.output_dir, "checkpoint-{}".format(step)
                #   )

                #   os.makedirs(output_dir_current, exist_ok=True)

                #   # Take care of distributed/parallel training
                #   model_to_save = (
                #       self.model.module if hasattr(self.model, "module") else self.model
                #   )
                #   model_to_save.save_pretrained(output_dir_current)
                #   self.tokenizer.save_pretrained(output_dir_current)

                # Calculate the average loss over all of the batches.
            avg_train_loss_g = tr_g_loss / len(self.train_dataloader)
            avg_train_loss_d = tr_d_loss / len(self.train_dataloader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss generetor: {0:.3f}".format(avg_train_loss_g))
            print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #     TEST ON THE EVALUATION DATASET
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our test set.
            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()  # maybe redundant
            self.discriminator.eval()
            self.generator.eval()

            # Tracking variables
            total_test_accuracy = 0

            total_test_loss = 0
            nb_test_steps = 0

            all_preds = []
            all_labels_ids = []

            # loss
            nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

            # Evaluate data for one epoch
            for batch in self.val_dataloader:
                # Unpack this training batch from our dataloader.
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    model_outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                    hidden_states = model_outputs[-1]
                    _, logits, probs = self.discriminator(hidden_states)
                    ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
                    filtered_logits = logits[:, 0:-1]
                    # Accumulate the test loss.
                    total_test_loss += nll_loss(filtered_logits, b_labels)

                # Accumulate the predictions and the input labels
                _, preds = torch.max(filtered_logits, 1)
                all_preds += preds.detach().cpu()
                all_labels_ids += b_labels.detach().cpu()

            # Report the final accuracy for this validation run.
            all_preds = torch.stack(all_preds).numpy()
            all_labels_ids = torch.stack(all_labels_ids).numpy()
            # print(all_preds)
            # print(all_labels_ids)
            # test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
            test_accuracy = accuracy_score(all_labels_ids, all_preds)
            print("  Accuracy: {0:.3f}".format(test_accuracy))

            precision = precision_score(all_labels_ids, all_preds, average='macro')
            recall = recall_score(all_labels_ids, all_preds, average='macro')
            f1 = f1_score(all_labels_ids, all_preds, average='macro')

            # Calculate the average loss over all of the batches.
            avg_test_loss = total_test_loss / len(self.val_dataloader)
            avg_test_loss = avg_test_loss.item()

            # Measure how long the validation run took.
            test_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.3f}".format(avg_test_loss))
            print("  Validation took: {:}".format(test_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss generator': avg_train_loss_g,
                    'Training Loss discriminator': avg_train_loss_d,
                    'Valid. Loss': avg_test_loss,
                    # TODO: rename test_accuracy to val_accuracy
                    'Valid. Accur.': test_accuracy,
                    'Training Time': training_time,
                    'Test Time': test_time,
                    'Valid. Precision': precision,
                    'Valid. Recall': recall,
                    'Valid. F1': f1
                }
            )

            if self.is_sweeping:
                wandb.log({
                    "training_loss_generator": avg_train_loss_g,
                    "training_loss_discriminator": avg_train_loss_d,
                    "training_loss": (avg_train_loss_g + avg_train_loss_d) / 2,
                    "validation_loss": avg_test_loss,
                    "validation_accuracy": test_accuracy,
                    'Valid. Precision': precision,
                    'Valid. Recall': recall,
                    'Valid. F1': f1
                })
        if not self.training_strategy == 'random_sampling_wandb':
            os.makedirs(self.output_dir, exist_ok=True)

            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model_to_save.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            torch.save(self.args, os.path.join(self.output_dir, "training_args.bin"))

            torch.save(self.generator.state_dict(), 'outputs/generator.pth')
            torch.save(self.discriminator.state_dict(), 'outputs/discriminator.pth')

    def test(self):
        # TODO: load model from best folder or wandb
        # TODO: self GAN model, not transformer model - for now were are saving transformer model
        model = self.model

        print("")
        print('Testing...')

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()  # maybe redundant
        self.discriminator.eval()
        self.generator.eval()

        total_test_loss = 0

        all_preds = []
        all_labels_ids = []

        # loss
        nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Evaluate data for one epoch
        for batch in self.test_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                model_outputs = model(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]
                _, logits, probs = self.discriminator(hidden_states)
                ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
                filtered_logits = logits[:, 0:-1]
                # Accumulate the test loss.
                total_test_loss += nll_loss(filtered_logits, b_labels)

            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        # print(all_preds)
        # print(all_labels_ids)
        # test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
        # print("  Accuracy: {0:.3f}".format(test_accuracy))
        test_accuracy = accuracy_score(all_labels_ids, all_preds)
        print("  Accuracy: {0:.3f}".format(test_accuracy))

        precision = precision_score(all_labels_ids, all_preds, average='macro')
        recall = recall_score(all_labels_ids, all_preds, average='macro')
        f1 = f1_score(all_labels_ids, all_preds, average='macro')
        # precision, recall, f1, support = score(all_labels_ids, all_preds)

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(self.test_dataloader)
        avg_test_loss = avg_test_loss.item()

        # Measure how long the validation run took.
        test_time = format_time(time.time() - t0)

        print("  Test Loss: {0:.3f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))

        if self.is_sweeping:
            wandb.log({
                "testing_loss": avg_test_loss,
                "testing_accuracy": test_accuracy,
                'testing_precision': precision,
                'testing_recall': recall,
                'testing_f1': f1
            })

    def save_wandb_model(self, run, model_name, output_dir, config):
        trained_model_artifact = wandb.Artifact(
            model_name, type="model",
            description="init training",
            metadata=dict(config))
        trained_model_artifact.add_dir(self.output_dir)
        run.log_artifact(trained_model_artifact)
        shutil.rmtree(self.output_dir)

    def wandb_train(self):
        if self.is_sweeping:
            with wandb.init(config=sweep_config_args) as run:
                self.train(wandb.config)
                # save trained model as artifact
                # TODO: pass model name, model dif as args
                self.test()
                # self.save_wandb_model(run, 'gan_bert_model', 'outputs/', wandb.config)
            wandb.finish()

    def train_with_sweep(self, output_dir):
        wandb.agent(self.sweep_id, self.wandb_train, count=self.args.agent_count)

    def best_model_train(self, params, output_dir, is_wandb=False):
        if self.evaluate:
            if self.is_sweeping and is_wandb:
                with wandb.init(config=params) as run:
                    self.evaluate_model(wandb.config)
                    # self.save_wandb_model(run, 'gan_bert_model', 'outputs/', wandb.config)
                wandb.finish()
            else:
                self.is_sweeping = False
                self.evaluate_model(params)
        else:
            if self.is_sweeping and is_wandb:
                with wandb.init(config=params) as run:
                    self.train(wandb.config)
                    # save trained model as artifact
                    # TODO: pass model name, model dif as args
                    self.test()
                    self.save_wandb_model(run, 'gan_bert_model', 'outputs/', wandb.config)
                wandb.finish()
            else:
                self.is_sweeping = False
                self.train(params)
                self.test()

    def random_sampling_with_wandb(self, output_dir, case_id):
        self.args.agent_count = 50  # TODO: refactor later
        self.case_id = case_id

        wandb.agent(self.sweep_id, self.random_sampling_with_wandb_sweep, count=self.args.agent_count)

    def random_sampling_with_wandb_sweep(self):
        if self.is_sweeping:
            # TODO: assign wandb configs based on param, do the same in constructor
            with wandb.init(config=self.random_sampling_args) as run:
                # reinitialise model for the next sets run
                self.model = self.model_default
                self.train(wandb.config)
                self.test()
            wandb.finish()

    def zero_shots_testing(self, params):
        with wandb.init(config=params) as run:
            self.evaluate_model(wandb.config)
            wandb.finish()

    def zero_shots_testing_with_wandb_sweep(self, output_dir, case_id):
        self.args.agent_count = 100  # TODO: refactor later
        self.case_id = case_id

        wandb.agent(self.sweep_id, self.zero_shot_random_sampling_with_wandb_sweep, count=self.args.agent_count)

    def zero_shot_random_sampling_with_wandb_sweep(self):
      if self.is_sweeping:
          # TODO: assign wandb configs based on param, do the same in constructor
          with wandb.init(config=self.random_sampling_args) as run:
              # reinitialise model for the next sets run
              self.model = self.model_default
              self.evaluate_model(wandb.config)
          wandb.finish()

    def _load_model_args(self, input_dir):
        args = GANBertModelArgs()
        args.load(input_dir)
        return args
