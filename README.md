### Greek OCR

Main goal of the project is to **Make it easier for greek byzantine researchers to transcribe the greek manuscripts** 

This is going to be a difficult task probably going to take months but the following outcomes can be expected : 

1. Novel Legal Manuscripts data curation $\to$ which can be made publically available
2. VLM Models & OCR models specifically for *greek* which can be used for various purposes
3. Platform for data annotation with expert in the loop to get accurate labels. 

This is a very novel work and can enhance the quality of work everywhere in the world. 

My **current priorites** are : 

1. Figuring out a good base line $\to$ which means experimenting with different providers, open source models to see what is the accuracy we are getting and using that baseline to : 
   1. Improve the upcoming models
   2. Make data annotation and labellling as easy as possible so that we can    
      1. Organize the data 
      2. Create a vast repository of annotated labels

---

*Thoughts* : I have been thinking of experimenting with googleOCR models via their APIs and using VLM open sourced for greek text identification in zero shot and also using openly available greek OCR models to figure out what is the current state. 

> [!IMPORTANT]
> Create a tool for **comparision** and **labelling** essentially data annotations. 

Have realised one thing that nothing is really working but there is till hope with 
1. `kraken` for segmentation and pre processing and using 
2. finetuned version of `TrOCR` for extracting the text from the segmented image. 

So problem of segmentation is not a problem. The pipeline would like to first binarize and then segment and using the segments we have to use the `TrOCR` models, the current segments are really really good. It uses the base model blla.mlmodel 

I think the `TrOCR` model is using the RoBERTa which is an english only architecture, so the current **cer** and **wer** stands at 0.97 and 1.33 which is really bad. Tried to replace the tokenizer with XLM-RoBERTa the **cer** and **wer** are 1.00 and 1.17 respectively. 

I have added the frontend capability to visualize the segments and my current **cer** is hovering around 0.75

--- 

Damm, this will definetly need fine tuning, relevant resources : 
1. [HuggingFace](https://discuss.huggingface.co/t/fine-tuning-trocr-on-new-language/58234)
2. [Github](https://github.com/huggingface/transformers/issues/19329)
3. [FineTuning](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)
4. [FineTuningOther](https://github.com/microsoft/unilm/issues/627)

