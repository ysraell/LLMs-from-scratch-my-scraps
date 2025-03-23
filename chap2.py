import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


# The author's approach stores repeated values massivelly.
class GPTDatasetLightStorev1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.ids = []

        token_ids = tokenizer.encode(txt)  # 1

        for i in range(0, len(token_ids) - max_length, stride):  # 2
            chunk = token_ids[i : i + max_length + 1]
            self.ids.append(torch.tensor(chunk))

    def __len__(self):  # 3
        return len(self.ids)

    def __getitem__(self, idx):  # 4
        return self.ids[idx][:-1], self.ids[idx][1:]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")  # 1
    # dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)   #2
    dataset = GPTDatasetLightStorev1(txt, tokenizer, max_length, stride)  # 2
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers  # 3  # 4
    )

    return dataloader
