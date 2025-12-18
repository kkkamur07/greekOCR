import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class CustomTrOCR(nn.Module):

    def __init__(self, processor, model_name="microsoft/trocr-base-stage1"):
        super().__init__()
        
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        original_vocab_size = self.model.config.decoder.vocab_size
        new_vocab_size = processor.tokenizer.vocab_size

        if original_vocab_size == new_vocab_size:
            self._update_token_ids(processor)

        else:
            self._resize_embeddings(processor, new_vocab_size)
        
        # Set generation config
        self._configure_generation()
    
    def _update_token_ids(self, processor):
        self.model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = processor.tokenizer.sep_token_id
        
        self.model.decoder.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        self.model.decoder.config.pad_token_id = processor.tokenizer.pad_token_id
        self.model.decoder.config.eos_token_id = processor.tokenizer.sep_token_id
    
    def _resize_embeddings(self, processor, new_vocab_size):
        self.model.decoder.resize_token_embeddings(new_vocab_size)
        
        self._update_token_ids(processor)
        
        self.model.config.vocab_size = new_vocab_size
        self.model.config.decoder.vocab_size = new_vocab_size
        self.model.decoder.config.vocab_size = new_vocab_size
    
    def _configure_generation(self):
        
        self.model.config.max_length = 64
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def generate(self, pixel_values, **kwargs):
        return self.model.generate(pixel_values, **kwargs)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        print(f"Model saved to {path}")
