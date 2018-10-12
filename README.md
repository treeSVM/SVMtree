# Tree-based SVM Classification
Here a python implementation of the Support Vector Machine Tree Optimal Class Partition (SVMTOCP) algorithm is provided [[Cohen et al. 2012]](https://link.springer.com/chapter/10.1007/978-3-642-33275-3_58) . It performs multiclass prediction based on a tree organization of SVM-based binary classifiers. The algorithm is based on the well known python libraries Numpy, Sklearn, NetworkX and Matplotlib.

## How to use it? 
It’s a very simple class and very easy to use. It only have 6 methods and we are going to explain them below. 

First of all, download the repository.

```
git clone https://github.com/treeSVM/treeSVM.git
```

Then move the folder to your project and import it in the python file.

```
from SVMTree import SVMTree
```

Now we have all set and ready to go. We just need to create a variable and assign SVMTree constructor.

```
svm = SVMTree()
``` 

### Methods

#### -Fit(X,Y)
This method allow you to train your svm tree. You need 2 parameters: 

X —> test data  
Y —> test target 

Both params need to be in Numpy array format. X may be a matrix, while Y need to be a single array of elements.

```
fit(xTrain, yTrain)
```

 #### -Predict(X)
This method gives you the prediction for a dataset. Again, it need to be in Numpy array format. 
 
```
predict(xTest)
```

#### -GetPerformance(Y)
This method pint the results obtained from comparing the prediction generated from your svm tree and the results passed in yTest as parameter. 

```
getPerformance(yTest) 
```

#### -PrintTree()
This method prints a plot of all of the tree’s nodes in its corresponding position. 

```
printTree() 
```

#### -SaveModel(path/name)
This methods saves your model in the destination ‘path’, creating a new binary file with 'modelName'. 

```
-saveModel(‘path/modelName’) 
```

#### -ReloadModel(path/name)
This method reload your model specified with 'modelName' from the origin ‘path’   

```
reloadModel(‘path/modelName’) 
```

## Performance Analysis
It was evaluated over 3 free available [data sets](https://archive.ics.uci.edu/ml/index.php) and compared against the One vs One SVM strategy available on SciPy.

### OVO/SVMTree
|DB|N|Classes|Vars|%ACC|%SV|C|%HMS|
| ---- | ---- | - | - | ----------| ------------ | ---------------- | --------- |
|Iris|150|3|3|98.33 / 100|17,5 / 38,30|100 / (0,01-100)|0 / 95,65|
|Glass|216|6|10|99.41 / 100|22,22 / 26.90|100 / (0,001-1-10-100)|7,89 / 52,17|
|Yeast|1479|10|9|60.49 / 100|80,26 / 13,39|130 / (1-10-100-130)|22,54 / 66,02|

%ACC: Accuracy, %SV: Percentage of Support Vector relative to the DB size, C: SVM constant constrain, %HMS: percentage of Hard Margin Solutions.

## Authors
Cassanelli Matías - Facultad de Ingeniería, Universidad Católica de Córdoba, Argentina (github user: MatiCassanelli)

Toledo Felipe -  Facultad de Ingeniería, Universidad Católica de Córdoba, Argentina (github user: felipetoledo4815)

Ing. Amiune Hernan @ GenerativeNetworks.com

Bioing. Fernandez Elmer Andrés (PhD) Bioscience Data Mining Group @ Centro de Investigación en Inmunología y Enfermedades Infecciosas, CONICET - Universidad Católica de Córdoba

Ing. Diego Arab Cohen.
