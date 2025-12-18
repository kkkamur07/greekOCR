from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrOCRProcessor, AutoTokenizer
from transformers import default_data_collator

from jiwer import wer, cer
import numpy as np
import torch

from ocr.data.TrOCRCollator import TrOCRDataCollator

from .data.estebanData import EstebanData
from .data.bibleDataset import BibleDataset
from .data.TrOCRCombinedDataset import TrOCRCombinedDataset

from .trOCR.model.trOCRModel import CustomTrOCR

#! TODO : Add argument to have pytorch_model.bin as well.
#! TODO : Need to be sure about the change of tokenizer in the processor as well, read about fine tuning trOCR.

GREEK_TOKENIZER = "xlm-roberta-base"
BASE_MODEL = "microsoft/trocr-base-stage1"
PROCESSOR = "microsoft/trocr-base-handwritten"

ANCIENT_GREEK_TOKENIZER = "pranaydeeps/Ancient-Greek-BERT"
BIBLE_DATA_PATH = "/home/krrish/Desktop/Programming/greekOCR/data/bibleTestament"
ESTEBAN_DATA_PATH = "/home/krrish/Desktop/Programming/greekOCR/data/estebanData"

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
    warmup_steps=500,
    weight_decay=0.1,
    learning_rate=1e-4,
    
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    
    output_dir="outputs/trocr-greek",
    logging_steps=1,
    save_steps=100,
    eval_steps=100,
    save_total_limit=2,
    num_train_epochs=10,
    
    gradient_accumulation_steps=4,
    metric_for_best_model="cer",
    greater_is_better=False,
    lr_scheduler_type="cosine",
    
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
    dataloader_persistent_workers=True,
)

# Metrics 
def compute_metricss(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    wer_score = wer(label_str, pred_str)
    cer_score = cer(label_str, pred_str)
    
    # Debug: Print some examples
    print("\n--- Sample Predictions ---")
    for i in range(min(3, len(pred_str))):
        print(f"Pred: {pred_str[i]}")
        print(f"True: {label_str[i]}")
        print()

    return {"wer": wer_score, "cer": cer_score}


def train() : 
    
    collator = TrOCRDataCollator(processor=processor)
    
    trainer = Seq2SeqTrainer(
        model=model.model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collator,
        compute_metrics=compute_metricss
    )

    trainer.train()
    trainer.save_model("outputs/trocr-greek/final_model")
    processor.save_pretrained("outputs/trocr-greek/final_model")
    
if __name__ == "__main__":
    train()