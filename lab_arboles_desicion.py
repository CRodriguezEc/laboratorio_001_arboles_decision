# %%
#   Importamos la libreria pandas para la lectura de archivos
import pandas as pd

#   Librerias para graficar los datos
import pandas.plotting as lbPlotting

#   Libreria para crear figuras en python
import matplotlib.pyplot as plt

#
import seaborn as sns

#   abrimos el archivo con el DataSet, y especificamos que cada registro esta delimintado por ";"
df = pd.read_csv( "Laboratorio_dataset_car.csv", delimiter = ";" )

# %%
#   Muestro un resumen del numero de filas y columnas del archivo (Dimensionalidad de nuestro DataSet) 
print( df.shape )

# %%
#   Muestro las tres(3) primeras lineas del archivo
print( df.head(10) )

# %%
#   Muestro de manera aleatoria 25 registros con la finalidad de ver un estado general de los archivos
print( df.sample(25) )

# %%
#   Muestro el resumen estadistico de los datos
print( df.describe() )

# %%
#   Muestro la estructura del DataSet
print( df.info() )

# %%
#   Identificamos si existen valores nulos
df.isnull().sum()

# %%
#   Distribucion por clases
print( df.groupby('class').size() )

# %%
#   Al momento de determinar la distribucion de valores de la "clase" se identifica que los valores de los atributos 
#   no se encuentran bien distribuidos, como es el caso del atributo "Unacc", lo cual se evidencia de mejor manera 
#   mediante el siguiente grafico.

sns.countplot( x = df["class"], data = df )

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in df.columns:
    df[i] = le.fit_transform( df[i] )

df.head()

# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# %%
X = df[df.columns[:-1]]
y = df['class']

#   Dividimos el dataset en 80% de los datos para entrenar y 20% para test   
X_train, X_validation, Y_train, Y_validation = train_test_split( X, y, test_size = 0.2, random_state = 1, shuffle = True )

# %%
#   Cargamos los algoritmos
models = []
models.append( ( 'LR', LogisticRegression( solver = 'liblinear', multi_class = 'ovr' ) ) )
models.append( ( 'CART', DecisionTreeClassifier() ) )

# %%
#   Evaluamos cada modelo por turno
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold( n_splits = 10, shuffle = True, random_state = 1 )
    cv_results = cross_val_score( model, X_train, Y_train, cv = kfold, scoring = 'accuracy' )
    results.append( cv_results )
    names.append(name)
    print( '%s: %f (%f)' % ( name, cv_results.mean(), cv_results.std() ) )



# %%
from matplotlib import pyplot

pyplot.boxplot( results, labels = names )
pyplot.title( 'Comparación de algoritmos' )
pyplot.show()

# %%
#   Realizamos predicciones con el dataset validacion
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier()
model.fit( X_train, Y_train )
predictions = model.predict( X_validation )

# %%
#   Evaluaciones las predicciones, en primer lugar la precision obtenida
print( accuracy_score( Y_validation, predictions ) )

# %%
#   Ahora la matriz de confusion
print( confusion_matrix( Y_validation, predictions ) )

# %%
#   Informe de clasificacion
print( classification_report( Y_validation, predictions ) )


