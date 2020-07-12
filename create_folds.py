import pandas as pd
from sklearn import model_selection
import os


if __name__ == "__main__":
    FILE_PATH = "" # TODO: Give file path here
    df = pd.read_csv(os.path.join(FILE_PATH, "train.csv"))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold_, (_,_) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, "kfold"] = fold_
    df.to_csv(os.path.join('data/', "train_folds.csv"), index=False)

    



    