import numpy as np
from joblib import load



long_hair = input("\n Does a person have long hair? [Yes or No]")
nose_wide = input("\n Does a person have wide nose? [Yes or No]")
nose_long = input("\n Does a person have long nose? [Yes or No]")
lips_thin  = input("\n Does a person have Thin lips ? [Yes or No]")
distance_nose_to_lip = input("\n Is there long distance between nose to lip [Yes or No]")
forehead_width = float(input("\n Enter the forehead width(in cm):"))
forehead_height = float(input("\n Enter the forehead height(in cm):"))



def standardize(X, mean, std):
    return (X-mean)/std

def label_encode(X):
    if(X.lower() == 'yes'):
        return 1
    elif(X.lower() == "no"):
        return 0

def standardization(forehead_height, forehead_width):
    height = standardize(forehead_height, 5.946311, 0.541268)
    width = standardize(forehead_width, 13.181484, 1.107128)
    return height, width

forehead_height, forehead_width = standardization(forehead_height, forehead_width)

long_hair = label_encode(long_hair)
nose_wide = label_encode(nose_wide)
nose_long = label_encode(nose_long)
lips_thin = label_encode(lips_thin)
distance_nose_to_lip = label_encode(distance_nose_to_lip)



test = [long_hair,	forehead_width,	forehead_height,	nose_wide,	nose_long,	lips_thin,	distance_nose_to_lip]
# test_2 = [0,	0.739389,	-1.009418,	0,	0,	1,	0]

X_test =  np.array(test).reshape(1,-1)
# X_test_1 = np.array(test_2).reshape(1,-1)
classifier = load('model.pkl')

y_pred = classifier.predict(X_test)
if(y_pred[0] == 0):
    print("You are Female.")
else:
    print("You are Male.")
# print(classifier.predict(X_test_1))