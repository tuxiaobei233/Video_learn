import json
import os
import random
d = open("datasets/data_video.json", "r").readlines()
for p in d:
    p = json.loads(p)
    if random.random() < 0.2:
        open("datasets/test.json",
             "a").write(json.dumps(p, ensure_ascii=False) + "\n")
    else:
        open("datasets/train.json",
             "a").write(json.dumps(p, ensure_ascii=False) + "\n")
