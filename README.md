# Font Recognition Project

## By Shoval Cohen 208748152

## Training

The project requires Python v3 to run.

To train the model on dataset, change the `file_path` variable at the file `train/train_model.py` to the `h5` dataset
file and run it.

```sh
python train/train_model.py
```

The trained model should be at the `resources` directory.

## Test and inference

To test the model on dataset, change the `file_path` variable at the file `test/test_model.py` to the `h5` test dataset
file and run it.

```sh
python test/test_model.py
```

The test should create the `results.csv` file.
