import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Base:
    def __init__(self):
       pass

    def preprocessing(self):

        self.df = pd.read_csv('diabetes.csv')

        self.X = self.df.drop('Outcome',axis=1).values
        self.y = self.df['Outcome'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=0)

        # Create tensors
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)
        self.y_train = torch.LongTensor(self.y_train)
        self.y_test = torch.LongTensor(self.y_test)


class ANN_Model(nn.Module):

    def __init__(self,input_features=8,hidden1=20,hidden2=20,output_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.out = nn.Linear(hidden2,output_features)

        
    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x

class Training(Base):
    def __init__(self):
        Base.__init__(self)
        self.epochs = 1000
        self.final_losses = []
        self.predictions = []

    def training(self):

        try:
            
            #Get the data ready
            self.preprocessing()

            torch.manual_seed(20)
            self.model = ANN_Model()
            print(self.model.parameters)

            #Backward Propagation and Optimizer
            try:
                self.loss_fn = nn.CrossEntropyLoss()
                self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)

                for i in range(self.epochs):
                    i= i+1
                    self.y_pred = self.model.forward(self.X_train)
                    self.loss = self.loss_fn(self.y_pred,self.y_train)
                    self.final_losses.append(self.loss)

                    if i % 10 == 1:
                        print(f'Epoch:{i} Loss:{self.loss}')
                    
                    self.optimizer.zero_grad()

                    self.loss.backward()

                    self.optimizer.step()

                if self.epochs == len(self.final_losses):
                    self.plotloss()
                    self.evaluate()

            except Exception as e:
                print(f'Child:{e}')
        except Exception as e:
            print(f'Master Expection:{e}')

        
    def plotloss(self):
        with torch.no_grad():
            plt.figure(figsize=(10,6))
            plt.plot(range(self.epochs) , self.final_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show(block=True)

    def evaluate(self):

        with torch.no_grad():
            for i, data in enumerate(self.X_test):
                self.y_evaluate = self.model(data)
                self.predictions.append(self.y_evaluate.argmax().item())

        score = accuracy_score(self.y_test,self.predictions)
        print(f'Accuracy Score:{score}')

        self.cm = confusion_matrix(self.y_test,self.predictions)
        plt.figure(figsize=(10,6))
        sns.heatmap(self.cm,annot=True)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show(block=True)

    def save_model(self):
         torch.save(self.model,'DiabetesClassifier.pt')

    def load_model(self):
        return torch.load('DiabetesClassifier.pt')

    def prediction(self):
        self.model1 = self.load_model()
         
        #Prediction of new data
        list1 = list(self.df.iloc[0,:-1])
        list1[1] = 130.0
        list1[3] = 40.0
        list1[4] =  0.0
        list1[5] = 25.6
        list1[7] = 45.0

        new_data = torch.tensor(list1)
        with torch.no_grad():
            self.new_pred_y = self.model1(new_data)
            print(self.new_pred_y.argmax().item())


def main():
    print('Hello')
    
    training = Training()
    training.training()
    training.save_model()

    training.prediction()



if __name__ == '__main__':
    main()



