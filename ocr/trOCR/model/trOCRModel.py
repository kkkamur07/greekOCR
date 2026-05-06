import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class CustomTrOCR(nn.Module):
    def __init__(self, processor, model_name="microsoft/trocr-small-stage1"):
        super().__init__()
        
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        self.model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = processor.tokenizer.sep_token_id
        
        self.model.decoder.config.vocab_size = processor.tokenizer.vocab_size
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        self.model.decoder.output_projection = nn.Linear(
            1024, 
            processor.tokenizer.vocab_size,
        )
        
        self.model.decoder.model.decoder.embed_tokens = nn.Embedding(
            processor.tokenizer.vocab_size, 
            1024, 
            padding_idx=self.model.config.pad_token_id
        )

        self.model.config.max_length = 64
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

    def forward(self, pixel_values, labels=None):
        
        return self.model(pixel_values=pixel_values, labels=labels)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
