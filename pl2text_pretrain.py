from tqdm import tqdm
import torch
import os
import json
import random
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import evaluate
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
random.seed(2022)

logger = logging.getLogger(__name__)
nltk.download('punkt')
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(
        default=None, metadata={"help": "Source language id for translation."})
    
    target_lang: str = field(
        default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

def convert_data(code_file, desc_file):
    code_data, desc_data = [], []
    with open(code_file, 'r') as f1, open(desc_file, 'r') as f2:
        temp_code_data = f1.read()
        temp_desc_data = f2.read()
    
    temp_code_data = temp_code_data.split('\n')
    temp_desc_data = temp_desc_data.split('\n')
    print('Converting data...')
    for i in tqdm(range(len(temp_code_data))):
        code, desc = temp_code_data[i].split(), temp_desc_data[i].split()
        if 200 >= len(code) >= 4 and 60 >= len(desc) >= 4:
            code_data.append(temp_code_data[i])
            desc_data.append(temp_desc_data[i].lower())
            
    assert len(code_data) == len(desc_data)
    idxs = random.sample(range(len(code_data)), 1000000)
    sub_set_code_data, sub_set_desc_data = [], []
    for idx in idxs:
        sub_set_code_data.append(code_data[idx])
        sub_set_desc_data.append(desc_data[idx])

    return sub_set_code_data, sub_set_desc_data

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses([
    "--train_file", "train_data_text_label.json",
    "--model_name_or_path", "uclanlp/plbart-base",
    "--dataset_name", "codesc",
    "--output_dir", "/localscratch/vjain312/pl2text",
    "--source_lang", "java",
    "--target_lang", "en_XX",
    "--ignore_pad_token_for_loss", "True",
    "--do_train", "True",
    "--learning_rate", "3e-4",
    "--generation_num_beams", "4",
    "--per_device_train_batch_size", "16",
    "--num_train_epochs", "3",
    "--save_total_limit", "1",
    "--overwrite_output_dir",
    "--predict_with_generate", "False",
    "--report_to", "wandb",
    "--run_name", "plbart",
    "--logging_steps", "20",
    "--save_strategy", "epoch",
    "--max_source_length", "300",
    "--max_target_length", "80",
    ])
set_seed(training_args.seed)

training_args.output_dir = os.path.join(training_args.output_dir, f"{data_args.dataset_name}_{data_args.source_lang}_{data_args.target_lang}_pretrain_mask_only_desc_1M")
training_args.run_name = f'{training_args.run_name}_{data_args.dataset_name}_{data_args.source_lang}_{data_args.target_lang}_pretrain_mask_only_desc_1M'

if not os.path.exists(training_args.output_dir):
    os.mkdir(training_args.output_dir)

if data_args.dataset_name == "codesearchnet":
    code_data_train, desc_data_train = convert_data(f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.code.pretrain.txt",
                                                    f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.desc.pretrain.txt")
elif data_args.dataset_name == "codesc":
    code_data_train, desc_data_train = convert_data(f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.code.pretrain.txt",
                                                    f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.desc.pretrain.txt")
with open(os.path.join(training_args.output_dir, 'train_data_text_label.json'), 'w') as openfile:
    for code, desc in zip(code_data_train, desc_data_train):
        temp = dict()
        temp["text"] = code
        temp["label"] = desc
        json.dump(temp, openfile)
        openfile.write('\n')

extension = "json"
raw_train_dataset = load_dataset(extension, data_files=os.path.join(training_args.output_dir, data_args.train_file), split="train")

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    src_lang=data_args.source_lang,
    tgt_lang=data_args.target_lang,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

tokenizer.add_tokens("java_en", special_tokens=True)
model.resize_token_embeddings(len(tokenizer))
tokenizer.lang_code_to_id["java_en"] = tokenizer.convert_tokens_to_ids("java_en")
tokenizer.src_lang = "java_en"
tokenizer.tgt_lang = "java_en"
model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("java_en")

tgt_text = raw_train_dataset[0]["label"]
print(f'Raw description: {tgt_text}')
labels = tokenizer(text_target=tgt_text, max_length=data_args.max_target_length, padding="max_length", return_tensors="pt").input_ids
print(f'Tokenized description: {tokenizer.convert_ids_to_tokens(list(labels[0]))}')

max_target_length = data_args.max_target_length
padding = "max_length" if data_args.pad_to_max_length else False

if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
    logger.warning(
        "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
        f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
    )

prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

def apply_masking(target):
    target_idxs = random.sample(range(0, len(target)), max(1, int(0.15 * len(target))))

    mask_token_idxs = random.sample(target_idxs, int(0.8 * len(target_idxs)))
    random_and_no_change_token_idxs = list(set(target_idxs) - set(mask_token_idxs))
    rtr_idxs = random.sample(random_and_no_change_token_idxs, int(0.5 * len(random_and_no_change_token_idxs)))
    token_idxs_left = list(set(range(len(target))) - set(target_idxs))

    for idx in mask_token_idxs:
        target[idx] = "<mask>"

    for idx in rtr_idxs:
        rand_token_idx = random.choice(token_idxs_left)
        target[idx] = target[rand_token_idx]

    return target

def apply_infilling(target):
    span_lengths = []
    total_to_be_masked = int(0.15 * len(target))
    num_masked = 0
    if total_to_be_masked == 0:
        idx = random.randint(0, len(target) - 1)
        target[idx] = "<mask>"
    else:
        while num_masked < total_to_be_masked:
            span_length = 0
            while span_length == 0:
                span_length = np.random.poisson(3.5)
            num_masked += span_length
            span_lengths.append(span_length)

        if sum(span_lengths) > int(0.45 * len(target)):
            target = apply_masking(target)
        else:
            infilling_mask = [0] * len(target)
            starts = random.sample(range(0, len(target)), len(span_lengths))
            for j in range(len(starts)):
                start = starts[j]
                end = min(len(infilling_mask), start + span_lengths[j])
                for k in range(start, end):
                    infilling_mask[k] = 1
            temp = []
            j = 0
            while j < len(infilling_mask):
                if infilling_mask[j] == 0:
                    temp.append(target[j])
                    j += 1
                else:
                    while j < len(infilling_mask) and infilling_mask[j] == 1:
                        j += 1
                    temp.append("<mask>")
            target = temp.copy()
    return target

def apply_token_deletion(target):
    target_idxs = random.sample(range(0, len(target)), max(1, int(0.15 * len(target))))

    delete_token_idxs = random.sample(target_idxs, max(1, int(0.8 * len(target_idxs))))
    rtr_idxs = list(set(target_idxs) - set(delete_token_idxs))
    to_be_deleted = [False] * len(target)
    for idx in delete_token_idxs:
        to_be_deleted[idx] = True

    for idx in rtr_idxs:
        rand_token_idx = random.choice(rtr_idxs)
        target[idx] = target[rand_token_idx]
    
    deleted_tokens_target = []
    for i in range(len(target)):
        if not to_be_deleted[i]:
            deleted_tokens_target.append(target[i])

    return deleted_tokens_target

def preprocess_function(examples):
    inputs, label_inputs, label_outputs = [], [], []

    for i in range(len(examples['text'])):
        input = examples['text'][i].split()
        target = examples['label'][i].split()
        target_noised = target.copy()
        task = random.randint(0, 1)
        if task == 0:
            target_noised = apply_masking(target_noised)
        elif task == 1:
            target_noised = apply_infilling(target_noised)
        else:
            target_noised = apply_token_deletion(target_noised) 
        
        model_input = input + ["</s>"] + target_noised
        # model_output = input + ["</s>"] + target
        model_output = target
        model_input = ' '.join(model_input)
        model_output = ' '.join(model_output)
        # print(f'Task: {task}')
        # print(f'Model input: {model_input}')
        # print(f'Model output: {model_output}')
        # print()
        # print('-----------------------------------------')
        # print()

        inputs.append(model_input)
        label_outputs.append(model_output)

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    
    model_label_outputs = tokenizer(text_target=label_outputs, max_length=data_args.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        model_label_outputs["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_label_outputs["input_ids"]
        ]
        
    model_inputs["labels"] = model_label_outputs["input_ids"]
    return model_inputs

column_names = raw_train_dataset.column_names
train_dataset = raw_train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=1,
    remove_columns=column_names,
    num_proc=data_args.preprocessing_num_workers,
    desc="Running tokenizer on train dataset",
)

ids = list(train_dataset[0]["input_ids"])
print(f'Example input ids: {ids}')
print(f'Example input tokens: {tokenizer.convert_ids_to_tokens(ids)}')
print()

label_ids = list(train_dataset[0]["labels"])
print(f'Example label ids: {label_ids}')
print(f'Example label tokens: {tokenizer.convert_ids_to_tokens(label_ids)}')
print()

model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)
print(f'Decoder start token id: {tokenizer.convert_ids_to_tokens(model.config.decoder_start_token_id)}')

# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)

# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=None,
)

checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
elif last_checkpoint is not None:
    checkpoint = last_checkpoint
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()
trainer.save_metrics("train", train_result.metrics)

print(train_result)