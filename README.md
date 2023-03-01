# Classification with KNN
Implementation of the KNN algorithm, validating it with the MNIST dataset, for the course Pattern Recognition.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 

Implementation of distance measurements:

 **- Manhattan**

 **- Euclidean**

Implementing training set reduction algorithms:

 **- Condensing**

 **- Editing**

## Run Locally


Clone the project
```bash
  git clone https://github.com/jean3P/ClassifierKNN.git
```

The mnist dataset (test.csv, train.csv) has to be in the directory:
```bash
  cd ClassifierKNN/resources/minst_small_knn
```

## Normal execution

After adding the data mnist, run the Main:
```bash
  run ClassifierKNN/src/Main_Knn_Without_Reducing_Dataset.py
```

## To use the reduction algorithms

The algorithm that prepares the reduction of the data is executed:
```bash
  run ClassifierKNN/src/Main_Reduce_Dataset.py
```

The new (reduced) data will be stored in:
```bash
  cd ClassifierKNN/resources/reduceDataset/condensing
```

```bash
  cd ClassifierKNN/resources/reduceDataset/editing
```

Finally, run the KNN algorithm:
```bash
  run ClassifierKNN/src/Main_Knn_With_Reduce_Dataset.py
```
## Statistics

- Ranking statistics for the entire training set (%):

| Distance  | K = 1  | k = 3  | k = 5  | k = 10 | k = 15 |
|:----------|:------:|:------:|:------:|:------:|:------:|
| Euclidean | 88.588 | 79.078 | 72.572 | 65.465 | 56.556 |
| Manhattan | 87.487 | 76.776 | 72.572 | 64.064 | 55.355 |

- Reduced training set sizes:

| Distance  | Condensing |  Editing   |
|:----------|:----------:|:----------:|
| Euclidean |   263760   |   612300   |
| Manhattan |   262190   |   606805   |

- Classification statistics using condensing (%):

| Distance  | K = 1  | k = 3  | k = 5  | k = 10 | k = 15 |
|:----------|:------:|:------:|:------:|:------:|:------:|
| Euclidean | 82.382 | 56.056 | 48.248 | 33.133 | 25.925 |
| Manhattan | 80.580 | 54.454 | 44.244 | 30.230 | 28.128 |

- Classification statistics using editing (%):

| Distance  | K = 1  | k = 3  | k = 5  | k = 10 | k = 15 |
|:----------|:------:|:------:|:------:|:------:|:------:|
| Euclidean | 86.586 | 77.477 | 73.973 | 63.963 | 57.257 |
| Manhattan | 83.883 | 74.274 | 69.969 | 62.862 | 57.657 |


## Author
- [PEREYRA PRINCIPE Jean Pool](https://github.com/jean3P)

## License

[MIT](https://choosealicense.com/licenses/mit/)


