import pickle as pkl
from numpy import pad
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam, AdamW, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
import os
import json
import itertools
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, concatenate_datasets

import evaluate
import statistics
import transformers
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
    EncoderDecoderModel,
    BertConfig,
    EncoderDecoderConfig,
    GPT2Config,
)
from transformers.trainer_utils import get_last_checkpoint
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# os.environ['WANDB_NOTEBOOK_NAME'] = 'pl2text'
# wandb.init(project="pl2text")

logger = logging.getLogger(__name__)
nltk.download('punkt')
# try:
#     nltk.data.find("tokenizers/punkt")
# except (LookupError, OSError):
#     if is_offline_mode():
#         raise LookupError(
#             "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
#         )
#     with FileLock(".lock") as lock:
#         nltk.download("punkt", quiet=True)

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
    
    return code_data, desc_data

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses([
    "--train_file", "train_data_text_label.json",
    "--validation_file", "val_data_text_label.json",
    "--test_file", "test_data_text_label.json",
    "--model_name_or_path", "microsoft/codebert-base",
    "--dataset_name", "codesc",
    "--output_dir", "/localscratch/vjain312/pl2text",
    "--source_lang", "java",
    "--target_lang", "en",
    "--ignore_pad_token_for_loss", "True",
    "--do_train", "True",
    "--do_eval", "True",
    "--do_predict", "True",
    "--learning_rate", "1e-4",
    "--generation_num_beams", "4",
    "--per_device_train_batch_size", "8",
    "--per_device_eval_batch_size", "8",
    "--num_train_epochs", "10",
    "--overwrite_output_dir",
    "--predict_with_generate", "True",
    "--report_to", "wandb",
    "--run_name", "codebert",
    "--logging_steps", "20",
    "--save_strategy", "epoch",
    "--evaluation_strategy", "steps",
    "--eval_steps", "4000",
    "--save_total_limit", "1",
    "--max_source_length", "250",
    "--max_target_length", "80",
    ])
set_seed(training_args.seed)

if 'checkpoint' in model_args.model_name_or_path:
    training_args.output_dir = os.path.join(training_args.output_dir, f'{data_args.dataset_name}_codebert')
    training_args.run_name = f'{training_args.run_name}_{data_args.dataset_name}_{data_args.source_lang}_{data_args.target_lang}'
else:
    training_args.output_dir = os.path.join(training_args.output_dir, f'{data_args.dataset_name}_codebert')
    training_args.run_name = f'{training_args.run_name}_{data_args.dataset_name}_{data_args.source_lang}_{data_args.target_lang}'

if not os.path.exists(training_args.output_dir):
    os.makedirs(training_args.output_dir)

if data_args.dataset_name == "codesearchnet":
    code_data_train, desc_data_train = convert_data(f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.code.train.txt",
                                                    f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.desc.train.txt")
elif data_args.dataset_name == "codesc":
    code_data_train, desc_data_train = convert_data(f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.code.train.txt",
                                                    f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.desc.train.txt")
with open(os.path.join(training_args.output_dir, 'train_data_text_label.json'), 'w') as openfile:
    for code, desc in zip(code_data_train, desc_data_train):
        temp = dict()
        temp["text"] = code
        temp["label"] = desc
        json.dump(temp, openfile)
        openfile.write('\n')

if data_args.dataset_name == "codesearchnet":
    code_data_val, desc_data_val = convert_data(f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.code.val.txt",
                                                    f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.desc.val.txt")
elif data_args.dataset_name == "codesc":
    code_data_val, desc_data_val = convert_data(f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.code.val.txt",
                                                    f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.desc.val.txt")
with open(os.path.join(training_args.output_dir, 'val_data_text_label.json'), 'w') as openfile:
    for code, desc in zip(code_data_val, desc_data_val):
        temp = dict()
        temp["text"] = code
        temp["label"] = desc
        json.dump(temp, openfile)
        openfile.write('\n')

if data_args.dataset_name == "codesearchnet":
    code_data_test, desc_data_test = convert_data(f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.code.test.txt",
                                                    f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.desc.test.txt")
elif data_args.dataset_name == "codesc":
    code_data_test, desc_data_test = convert_data(f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.code.test.txt",
                                                    f"/localscratch/vjain312/DL-project-data/{data_args.dataset_name}/{data_args.source_lang}.desc.test.txt")
with open(os.path.join(training_args.output_dir, 'test_data_text_label.json'), 'w') as openfile:
    for code, desc in zip(code_data_test, desc_data_test):
        temp = dict()
        temp["text"] = code
        temp["label"] = desc
        json.dump(temp, openfile)
        openfile.write('\n')

extension = "json"
raw_train_dataset = load_dataset(extension, data_files=os.path.join(training_args.output_dir, data_args.train_file), split="train")
raw_validation_dataset = load_dataset(extension, data_files=os.path.join(training_args.output_dir, data_args.validation_file), split="train")
raw_test_dataset = load_dataset(extension, data_files=os.path.join(training_args.output_dir, data_args.test_file), split="train")

tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=model_args.use_fast_tokenizer,
)

model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    model_args.model_name_or_path, 
    model_args.model_name_or_path, 
)

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

print()
src_text = raw_train_dataset[0]["text"]
print(f'Raw code: {src_text}')
inputs = tokenizer(src_text, max_length=data_args.max_source_length, padding="max_length", return_tensors="pt").input_ids
print(f'Tokenized code: {tokenizer.convert_ids_to_tokens(list(inputs[0]))}')
print()

tgt_text = raw_train_dataset[0]["label"]
print(f'Raw description: {tgt_text}')
labels = tokenizer(text_target=tgt_text, max_length=data_args.max_target_length, padding="max_length", return_tensors="pt").input_ids
print(f'Tokenized description: {tokenizer.convert_ids_to_tokens(list(labels[0]))}')
print()

max_target_length = data_args.max_target_length
padding = "max_length" if data_args.pad_to_max_length else False

if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
    logger.warning(
        "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
        f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
    )

prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

def preprocess_function(batch):
    inputs = tokenizer(batch['text'], max_length=data_args.max_source_length, padding="max_length", truncation=True)
    labels = tokenizer(batch['label'], max_length=data_args.max_target_length, padding="max_length", truncation=True)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = labels.input_ids
    batch["decoder_attention_mask"] = labels.attention_mask
    batch["labels"] = labels.input_ids.copy()
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    return batch

column_names = raw_train_dataset.column_names
train_dataset = raw_train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=1,
    remove_columns=column_names,
    num_proc=data_args.preprocessing_num_workers,
    desc="Running tokenizer on train dataset",
)

val_dataset = raw_validation_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=1,
    remove_columns=column_names,
    num_proc=data_args.preprocessing_num_workers,
    desc="Running tokenizer on validation dataset"
)

test_dataset = raw_test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=1,
    remove_columns=column_names,
    num_proc=data_args.preprocessing_num_workers,
    desc="Running tokenizer on test dataset",
)

print('Vocab Size: ', model.config.encoder.vocab_size)

ids = list(train_dataset[0]["input_ids"])
print(f'Example input ids: {ids}')
print(f'Example input tokens: {tokenizer.convert_ids_to_tokens(ids)}')
print()

label_ids = list(train_dataset[0]["decoder_input_ids"])
print(f'Example label ids: {label_ids}')
print(f'Example label tokens: {tokenizer.convert_ids_to_tokens(label_ids)}')
print()

sacrebleu = evaluate.load("sacrebleu")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    label_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    result = {}
    
    result["sacrebleu"] = sacrebleu.compute(predictions=pred_str, references=label_str)
    result["bleu"] = bleu.compute(predictions=pred_str, references=label_str)
    result["bleu_1"] = bleu.compute(predictions=pred_str, references=label_str, max_order=1)
    result["bleu_2"] = bleu.compute(predictions=pred_str, references=label_str, max_order=2)
    result["bleu_3"] = bleu.compute(predictions=pred_str, references=label_str, max_order=3)
    result["bleu_4"] = bleu.compute(predictions=pred_str, references=label_str, max_order=4)
    result["rouge"] = rouge.compute(predictions=pred_str, references=label_str)
    # result = {k: round(v, 4) for k, v in result.items()}
    return result

# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=val_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    # data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
)

if training_args.do_train:
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_metrics("train", train_result.metrics)
    print(train_result)


if training_args.do_predict:
    predict_results = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=training_args.generation_max_length, num_beams=training_args.generation_num_beams)
    trainer.save_metrics("predict", predict_results.metrics)
    print(predict_results)

if training_args.predict_with_generate:
    predictions = tokenizer.batch_decode(
        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    predictions = [pred.strip() for pred in predictions]
    output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        writer.write("\n".join(predictions))