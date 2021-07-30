# LRGCPND

This is an example implementation of the LRGCPND model.

## Environment Settings

package | version
:--:|:--:
python          |   3.8
numpy           |   1.20
pandas          |   1.2
scipy           |   1.6
scikit-learn    |   0.24
torch           |   1.8
cudatoolkit     |   10.2

## Usage

Run ``python run.py -h/--help`` for more detailed usage:

```bash
usage: run.py [-h] [--n_num N_NUM] [--d_num D_NUM] [-K K] [-S S] [-r REG] [-l LR] [-e EPOCHS] [-b BATCH] [-f FOLD] [-t TIME] [--save_models]
              [--have_trained]

optional arguments:
  --n_num N_NUM         Number of ncRNAs.
  --d_num D_NUM         Number of drugs.
  -K K                  Depth of layers.
  -r REG, --reg REG     Coefficient of L2 regularization.
  -l LR, --lr LR        Initial learning rate.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train.
  -b BATCH, --batch BATCH
                        Batch size to train.
  -f FOLD, --fold FOLD  Number of folds for cross validation.
  -t TIME, --time TIME  Timestamp in milliseconds for training.
  --save_models         Save trained models (true or false).
  --have_trained        Have trained models (true or false).
```

### Generate samples for k-fold CV (necessary for the first time)

The randomly generated triples will be saved in ``/data/samples``.

### Run (train and test)

``python run.py``

or if you'd like to save the models:

``python run.py --save_models``

We use timestamps to mark models and corresponding results.

### Evaluate (load and test)

``python run.py --have_trained -t {str_time}``

where {str_time} is the timestamp of your trained model.

## Cite
