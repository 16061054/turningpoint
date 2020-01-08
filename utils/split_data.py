import numpy as np


def split(X, y,SEED=2000, TRAIN_RATIO=0.8, VALIDATION_RATIO=0.1):
    """split the dataset."""
    # set seed
    np.random.seed(SEED)

    # permutation indices.
    dim_y_row = len(y)
    #shuffle_indices = np.random.permutation(np.arange(dim_y_row))
    shuffle_indices = np.arange(dim_y_row) # 保持顺序
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # split the data to train, validation and test.
    X_train = X_shuffled[: int(dim_y_row * TRAIN_RATIO)]
    y_train = y_shuffled[: int(dim_y_row * TRAIN_RATIO)]
    X_validation = X_shuffled[
        int(dim_y_row * TRAIN_RATIO):
        int(dim_y_row * TRAIN_RATIO + dim_y_row * VALIDATION_RATIO)]
    y_validation = y_shuffled[
        int(dim_y_row * TRAIN_RATIO):
        int(dim_y_row * TRAIN_RATIO + dim_y_row * VALIDATION_RATIO)]
    X_test = X_shuffled[
        int(dim_y_row * TRAIN_RATIO + dim_y_row * VALIDATION_RATIO):]
    y_test = y_shuffled[
        int(dim_y_row * TRAIN_RATIO + dim_y_row * VALIDATION_RATIO):]

    splited_data = {
        "train_data": X_train,
        "train_labels": y_train,
        "validation_data": X_validation,
        "validation_labels": y_validation,
        "test_data": X_test,
        "test_labels": y_test
    }

    ctp_train = np.sum((y_train != 0).astype(int))
    ctp_val = np.sum((y_validation != 0).astype(int))
    ctp_test = np.sum((y_test != 0).astype(int))

    print("ctp: train/validation/test split: {}/{}/{}".format(ctp_train, ctp_val, ctp_test))
    print("all: train/validation/test split: {}/{}/{}".format(splited_data["train_data"].shape, splited_data["validation_data"].shape, splited_data["test_data"].shape))
    return splited_data