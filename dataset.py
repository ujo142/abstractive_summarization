from torch.utils.data import Dataset
from tqdm import tqdm
import csv


class CsvDataset(Dataset):
    def __init__(self, data_path, max_len=None, append_prefix=None):
        if append_prefix == None:
            append_prefix = ""
        self.append_prefix = append_prefix
        self.samples = list()
        with open(data_path, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            for line in tqdm(csv_reader):
                self.samples.append(line)
                if max_len != None and len(self.samples) > max_len:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {}
        sample["id"] = self.samples[idx][0]
        sample["input"] = self.append_prefix + str(self.samples[idx][2])
        sample["target"] = self.samples[idx][1]

        return sample
