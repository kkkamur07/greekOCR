from ocr.data.bibleDataset import BibleDataset
import transformers
from jiwer import wer, cer
from tqdm import tqdm

from transformers import XLMRobertaTokenizer

def main(root : str) : 
    processor = transformers.TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    
    tokenizer2 = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    processor.tokenizer = tokenizer2
    
    model = transformers.VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

    dataset = BibleDataset(root=root, split="test", binarize=True)
    
    print(f"Dataset size: {len(dataset)}")

    total_wer = 0.0
    total_cer = 0.0
    n = len(dataset)

    for i in tqdm(range(n)):
        image, label = dataset[i]
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        total_wer += wer(label, generated_text)
        total_cer += cer(label, generated_text)

    avg_wer = total_wer / n
    avg_cer = total_cer / n

    return avg_wer, avg_cer

if __name__ == "__main__" :
    root = "/Users/krishuagarwal/Desktop/Programming/python/greek-ocr/data/labelledData"
    wer, cer = main(root)
    
    print(f"Average WER: {wer}")
    print(f"Average CER: {cer}")