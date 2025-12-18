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


#### Date : 15 Nov 2025

*Thoughts* : I have realised one thing, without finetuning & data we are not going to make much progress. Probably we should fine tune `trOCR` or `deepSeekOCR`. 


