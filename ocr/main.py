from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrOCRProcessor, AutoTokenizer
from transformers import default_data_collator

from jiwer import wer, cer
import numpy as np
import torch

from .data.estebanData import EstebanData
from .data.bibleDataset import BibleDataset
from .data.TrOCRCombinedDataset import TrOCRCombinedDataset

from .trOCR.model.trOCRModel import CustomTrOCR


GREEK_TOKENIZER = "xlm-roberta-base"
BASE_MODEL = "microsoft/trocr-base-stage1"
PROCESSOR = "microsoft/trocr-base-handwritten"

BIBLE_DATA_PATH = "/Users/krishuagarwal/Desktop/Programming/python/greek-ocr/data/labelledData"
ESTEBAN_DATA_PATH = "/Users/krishuagarwal/Desktop/Programming/python/greek-ocr/data/estebanData"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Processor
processor = TrOCRProcessor.from_pretrained(PROCESSOR)
xlmTokenizer = AutoTokenizer.from_pretrained(GREEK_TOKENIZER)
processor.tokenizer = xlmTokenizer

# Datasets
train_bible = BibleDataset(root=BIBLE_DATA_PATH, split="train")
train_esteban = EstebanData(root=ESTEBAN_DATA_PATH, split="train")
train_data = TrOCRCombinedDataset(datasets=[train_bible, train_esteban], processor=processor)

test_bible = BibleDataset(root=BIBLE_DATA_PATH, split="test")
test_esteban = EstebanData(root=ESTEBAN_DATA_PATH, split="test")
test_data = TrOCRCombinedDataset(datasets=[test_bible, test_esteban], processor=processor)

model = CustomTrOCR(processor=processor, model_name=BASE_MODEL)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",
    warmup_steps=1000,
    learning_rate=3e-4,
    weight_decay=0.1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=False,
    output_dir="outputs/trocr-greek",
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    num_train_epochs=10,
)

# Metrics 
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    wer_score = wer(label_str, pred_str)
    cer_score = cer(label_str, pred_str)

    return {"wer": wer_score, "cer": cer_score}


def train() : 
    trainer = Seq2SeqTrainer(
        model=model.model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
if __name__ == "__main__":
    train()