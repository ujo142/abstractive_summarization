import pytorch_lightning as pl
from dataloader import T5Dataloader
from dataset import CsvDataset
from model import T5
import torch
import warnings

from transformers import T5Tokenizer, T5ForConditionalGeneration

def main():
    warnings.filterwarnings("ignore")
    print("Preparing dataloaders...")
    loader = T5Dataloader(
           train_data_path="/Users/ben/python_projects/summarization/abstractive_summarization/news_summaries_prepared.csv",
           ratio=0.8,
           batch_size=4, 
           workers=0, 
           prefix="Summarize: ")    

    train_data, val_data = loader.get_dataloaders()


    print("Initializing the model...")
    model = T5(
          learning_rate=4e-5,
          multiply_lr_step=0.7,
          warmup_steps=100.0,
          model_path="t5-small",
          model_save_directory="t5-small.pkl",
          max_source_length=512,
          max_target_length=128,
          model_load_directory=None)
           

    print("Training the model...")
    trainer = pl.Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            log_every_n_steps=5,
            enable_progress_bar=True)
              
    trainer.fit(model, train_data, val_data)

   #model = model.eval()
   # output = model.model.generate("Summarize: "+"Hi iam joseph i came from norway!")
    #print(output)



    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    

    task_prefix = "Summarize: "
    sentences = ["It's just a big term for me."" He added that apart from low blood pressure, he isn't facing any health issues and is feeling perfectly fine now. Kapil also clarified he has no stress and pressure regarding The Kapil Sharma Show's TRP.","Comedian and TV?show host Kapil Sharma has finally spoken on the rumours of his depression. In an interview to Dainik Bhaskar, Kapil cleared the air surrounding his ill health and said that he merely has blood pressure problems, not depression.According to several rumours floating around since past few months, Kapil has been under a lot of stress due to the falling TRPs of his show, The Kapil Sharma Show. ?It (depression) is just a big term for me (laughs). Apart from low blood pressure, I am not at all facing any health issues. I have started taking care of my diet and body. I am perfectly fine now,? he said in the interview.Mumbai: Comedian Kapil Sharma Bollywood actors Riteish Deshmukh and Vivek Oberoi during the promotion of film Bank Chor on the sets of The Kapil Sharma Show in Mumbai, on June 8, 2017. (Photo: IANS)"]

    inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

    output_sequences = model.model.generate(
                     input_ids=inputs["input_ids"],
                     attention_mask=inputs["attention_mask"],
                     do_sample=False)

    print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
if __name__ == "__main__":
    main()