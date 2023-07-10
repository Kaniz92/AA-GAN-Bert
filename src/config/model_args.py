import json
import os
from dataclasses import dataclass


@dataclass
class ModelArgs:
    max_seq_length: int = 512
    noise_size: int = 100
    apply_balance: bool = True
    learning_rate_discriminator: float = 5e-5
    learning_rate_generator: float = 5e-5
    epsilon: float = 1e-8
    multi_gpu: bool = True
    apply_scheduler: bool = True
    print_each_n_step: int = 10
    agent_count: int = 1
    model_name: str = "bert-base-cased"
    manual_seed: int = 42
    output_dir: str = 'outputs/'

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)


@dataclass
class GANBertModelArgs(ModelArgs):
    model_class: str = 'GANBertModel'
