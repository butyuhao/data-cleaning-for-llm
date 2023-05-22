import json
from datasets import load_dataset, Dataset

class MyDialogue(Dataset):
    def __init__(self, mode: str, cache_dir: str = None) -> None:
        self.mode = mode
        self.rows = []
        with open(cache_dir,"r",encoding="utf-8") as f:
            self.rows = f.readlines()
        def get(x):
            l = []
            x = x["thread"]
            while True:
                l.append(x["text"])
                if len(x["replies"])==0:
                    break
                x = x["replies"][0]
            return l
        for i in range(len(self.rows)):
            self.rows[i] = get(json.loads(self.rows[i]))
            # print(self.rows[i])


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        dialogue: list = self.rows[index]
        if self.mode == "sft":
            return dialogue
        elif self.mode == "rl":
            return tuple(dialogue[:-1])