import sys
import csv
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

def main(file):
    X=np.empty([0,2])
    Y=np.empty([0,1])
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            X=np.vstack([X,[1,int(row['year'])]])
            Y=np.vstack([Y,[int(row['days'])]])
    
    Z=np.dot(X.T,X)
    I=np.linalg.inv(Z)
    PI=np.dot(I,X.T)
    hat_beta=np.dot(PI,Y)
    
    y_hat=hat_beta[0]+hat_beta[1]*2021
    x_star= -hat_beta[0]/hat_beta[1]
    
    
    df=pd.read_csv(file)
    plt.plot(df["year"],df["days"])
    plt.xlabel('Year')
    plt.ylabel('Number of frozen days')
    plt.savefig("plot.png")
    
    print("Q3a:")
    print(X)
    
    print("Q3b:")
    print(Y.T)
    
    print("Q3c:")
    print(Z)
    
    print("Q3d:")
    print(I)
    
    print("Q3e:")
    print(PI)
    
    print("Q3f:")
    print(hat_beta.T)
    
    print("Q4: "+str(y_hat[0]))
    
    print("Q5a: >")
    print("Q5b: For each increase in a year, the days covered in ice increases by 85.59")
    
    print("Q6a: "+str(x_star[0]))
    
    print("Q6b: Given the general slight downward slope of the data (even with the erratic peaks), it is reasonable by the year 2456 will there be no more days where Lake Mendota is covered in ice given the increasing general warmth of the planet due to global warming ")
    
if __name__ == "__main__":
    s1=sys.argv[1]
    main(s1)

    

