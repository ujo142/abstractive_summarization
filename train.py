import pytorch_lightning as pl
from dataloader import T5Dataloader
from dataset import CsvDataset
from model import T5
import torch
import warnings

from transformers import T5Tokenizer, T5ForConditionalGeneration

def main():
    print("Checking acceleration...")
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("Running on the GPU")
    else:
        dev = torch.device("cpu")
        print("Running on the CPU")  

    warnings.filterwarnings("ignore")
    print("Preparing dataloaders...")
    loader = T5Dataloader(
           train_data_path="/Users/ben/python_projects/summarization/abstractive_summarization/news_summaries_2.csv",
           ratio=0.8,
           batch_size=32, 
           workers=0, 
           prefix="Summarize: ")    

    train_data, val_data = loader.get_dataloaders()


    print("Initializing the model...")
    model = T5(
          learning_rate=6e-5,
          multiply_lr_step=0.7,
          warmup_steps=100.0,
          model_path="t5-small",
          model_save_directory="t5-small.pkl",
          max_source_length=512,
          max_target_length=128,
          model_load_directory=None,
          dev=dev)
          
           

    print("Training the model...")
    trainer = pl.Trainer(
            max_epochs=2,
            accelerator='mps',
            devices=1,
            log_every_n_steps=5,
            enable_progress_bar=True)
              
    trainer.fit(model, train_data, val_data)

   #model = model.eval()
   # output = model.model.generate("Summarize: "+"Hi iam joseph i came from norway!")
    #print(output)



    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    

    task_prefix = "Summarize: "
    sentences = ["Sitting by the bed, holding my hand, you think my mind is fighting against the decision of my body to quit life’s game. My eyes are closed, but I sense your will through the fingers laced tightly around my own. Tenderness is a force and you stake my claim to life through the insistent pressure of your hand. How it has grown over these long years from its immaculate small perfection to this manifestation of adult capability: greeting strangers, shaking on deals, carrying children of your own. From the first moment, holding tight to my little finger in the hush of the darkened hospital room, it wanted to latch onto me and the world. You needed reassuring then; you do now. "]

    inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

    output_sequences = model.model.generate(
                     input_ids=inputs["input_ids"],
                     attention_mask=inputs["attention_mask"],
                     do_sample=False)

    print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
if __name__ == "__main__":
    main()