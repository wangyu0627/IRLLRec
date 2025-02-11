# IRLLRec
Intent Representation Learning with Large Language Model for Recommendation

We are organizing the code as quickly as possible. If there are any issues, please stay tuned to our anonymous GitHub.

You can download semantic embedding files in the following datasets:

**Amazon-book**/ **Yelp**/ **Amazon-movie** [[Google Drive]](https://drive.google.com/drive/folders/1eeaEHLJFYH9Kc_-Y81473tMiAPaGcQVV?usp=drive_link/view)

Each dataset consists of a training set, a validation set, and a test set. During the training process, we utilize the validation set to determine when to stop the training in order to prevent overfitting.

```plaintext
- book (yelp/movie)
|--- trn_mat.pkl # training set (sparse matrix)
|--- val_mat.pkl # validation set (sparse matrix)
|--- tst_mat.pkl # test set (sparse matrix)
|--- usr_emb_np.pkl # user text embeddings
|--- itm_emb_np.pkl # item text embeddings
|--- user_intent_emb_3.pkl # user intent embeddings
|--- item_intent_emb_3.pkl # item intent embeddings
```

# 🚀 Examples to run the codes

The command to evaluate the backbone models and RLMRec is as follows.

- **Backbone**
  ```bash
  python encoder/train_encoder.py --model {model_name} --dataset {dataset} --cuda 0

- **RLMRec**
  ```bash
  python encoder/train_encoder.py --model {model_name}_plus --dataset {dataset} --cuda 0
  ```
  ```bash
  python encoder/train_encoder.py --model {model_name}_gene --dataset {dataset} --cuda 0

- **IRLLRec**
  ```bash
  python encoder/train_encoder.py --model {model_name}_int --dataset {dataset} --cuda 0

Hyperparameters:

The hyperparameters of each model are stored in encoder/config/modelconf (obtained by grid-search).


