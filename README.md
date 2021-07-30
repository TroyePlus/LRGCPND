# LRGCPND

This is an example implementation of the LRGCPND model.

## Environment

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

### generate samples for k-fold CV (necessary for the first time)

The randomly generated triples will be saved in ``/data/sample``.

### run (train and test)

``python run.py``

or if you'd like to save the models:

``python run.py --save_models``

### load and test

``python run.py --have_trained -t {str_time}``

## Cite
