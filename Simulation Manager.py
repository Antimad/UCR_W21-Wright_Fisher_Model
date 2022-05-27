from multiprocessing import Pool
import os
import WF
import Inference
import pandas as pd
"""
params = {
    "File Name": 1,
    "Generations": 400,
    "Selection": 0.05,
    "Sequence Length": 10,
    "Mutation Rate": 1e-3
}
"""
# For Selection
folder = "Data"
Selection_Data = [dict({"File Name": os.path.join(folder, f), "Generations": 400, "Selection": 0.05, "Sequence Length": 10,
                  "Mutation Rate": 1e-3}) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# For WF
WF_data = [dict({"File Name": x, "Generations": 400, "Selection": 0.02, "# of Selection": 10, "Sequence Length": 50,
                 "Mutation Rate": 2e-3}) for x in range(1, 101)]


def wright_fisher(inputs):
    if __name__ == '__main__':
        pool = Pool(13)  # 13 for ~95% usage
        outputs = pool.map(WF.main, inputs)
        return outputs


def selection_inference():
    if __name__ == "__main__":
        folder = "Data"
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        pool = Pool(13)
        outputs = pool.map(Inference.main, files)
        data = outputs
        df = pd.DataFrame(data)
        df.to_csv("Selection Inference Results.csv", index=False)


wright_fisher(WF_data)
