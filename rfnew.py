import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

from astropy.table import Table
from astropy.table import Column
import sys
import os
import argparse
from joblib import dump, load
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report



#Load a csv or a vot file
def loadFile(fileName, isCsv):
 if(isCsv):
  return pd.read_csv(fileName)
 else:
  return Table.read(fileName, format='votable')


parser = argparse.ArgumentParser()
parser.add_argument('-train', nargs='?', const='set', help='Train a Random Forest with a dataset in format csv, the default file is "set"')
parser.add_argument('-save', nargs='?', const='model', help='Save the trained model, if a name is not specified it will be "model"')
parser.add_argument('-test', nargs='+', metavar='Files to test', help='List of files to test the model')
parser.add_argument('-loadmodel', nargs='?', const='model', help='Load a trained model to test without train again, if a name is not specified it will be "model"')
parser.add_argument('-labels', nargs='+', help='[Optional in -test, Max 3 chars] List of strings to labels the prediction. It must be in ondered to match with the dataset used to train')
parser.add_argument('-v', action='store_true', help='Display more information')
args = parser.parse_args()
if(args.train is None and args.test is not None and args.loadmodel is None):
 parser.error('To use -test without -train is requiered specify -loadmodel')
if(args.train is None and args.save):
 parser.error('-save requiered use -train')

rf = RandomForestClassifier()
if(args.train is not None):
 #Reading file
 features = loadFile(args.train+'.csv', True)


 labels = np.array(features['label'])
 features= features.drop('label', axis = 1)
 feature_list = list(features.columns)
 features = np.array(features)

 train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33, random_state = 42, stratify=labels)

 rf = RandomForestClassifier(n_estimators=2000, max_features=7, max_leaf_nodes=None, random_state=42)
 rf.fit(train_features, train_labels);

 predictions = rf.predict(test_features)
 # Calculate the absolute errors
 errors = abs(predictions - test_labels)
 # Print out the mean absolute error (mae)
 #if(args.v):
  #print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

 # Calculate mean absolute percentage error (MAPE)
 mape = 100 * (errors / test_labels)
 # Calculate and display accuracy
 accuracy = 100 - np.mean(mape)
 print('Accuracy MAPE:', round(accuracy, 4))

 # Pull out one tree from the forest
 #tree = rf.estimators_[5]

 # Export the image to a dot file
 #export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
 # Use dot file to create a graph
 #(graph, ) = pydot.graph_from_dot_file('tree.dot')
 # Write graph to a png file
 #graph.write_png('tree.png')

 if(args.v):
  cm=confusion_matrix(test_labels, predictions)
  print ("Test Accuracy  :: ", accuracy_score(test_labels, predictions))
  print (" Confusion matrix ", confusion_matrix(test_labels, predictions))

  print('SENSIVITY',cm[1,1]/(cm[1,0]+cm[1,1]))
  print("precision_score", precision_score(test_labels, predictions))
  print('SPECIFITY ',cm[0,0]/(cm[0,0]+cm[0,1]) )
  print("f1_score", f1_score(test_labels, predictions))
  print("Cohen's kappa", cohen_kappa_score(test_labels, predictions))
  print('matthews_corrcoef ',matthews_corrcoef(test_labels, predictions))

  print(classification_report(test_labels, predictions, labels=[1, 2], digits=4))
  #print(classification_report(test_labels, predictions, labels=[1, 2, 3], digits=4))


  #########
  # Get numerical feature importances
  importances = list(rf.feature_importances_)
  # List of tuples with variable and importance
  feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
  # Sort the feature importances by most important first
  feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
  # Print out the feature and importances
  [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
  ##########

 ###Save the current Random Forest in a file
 if(args.save is not None):
  dump(rf, args.save+'.joblib')

if(args.test is not None):

 #Load a Random Forest from a file
 if(args.loadmodel is not None):
  rf=load(args.loadmodel+'.joblib')
 #Reading file to test the tree
 for fileToTest in range(len(args.test)):
  mainTable=loadFile(args.test[fileToTest]+'.vot', False)
  #print ('predictions')

  #Adding a new column to save the predictions
  newColumn=Column(data=['   ']*len(mainTable), name='prediction' )
  mainTable.add_column(newColumn)
  print(shape(mainTable))


  #Evaluate each star and save prediction
  for star in range(len(mainTable)):
   prediction=rf.predict([[mainTable[star]['bp_rp'], mainTable[star]['g_rp'], mainTable[star]['bp_g'], mainTable[star]['g_abs_mag'], mainTable[star]['teff_val'], mainTable[star]['j_h'], mainTable[star]['h_k']]]) #, mainTable[star]['b_v']]])
   prediction[0]=prediction[0]/2.0

   #Change the tag for a label
   if(args.labels is not None):
    mainTable[star]['prediction']=args.labels[int(prediction[0]-1)]
   else:
    mainTable[star]['prediction']=prediction[0]

  #Save the predictions in a new file
  if(os.path.exists(sys.argv[fileToTest]+'_predictions.vot')):
   os.remove(sys.argv[fileToTest]+'_predictions.vot')
  mainTable.write(sys.argv[fileToTest]+'_predictions.vot', format='votable')

if(args.train is not None or args.test is not None):
 print("process-finished")
