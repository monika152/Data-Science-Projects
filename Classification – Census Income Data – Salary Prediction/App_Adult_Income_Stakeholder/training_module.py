


#Train and Prediction the Model
class Train_predict:
    
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        
        from sklearn import tree
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(self.X, self.y)
        accuracy = self.model.score(X, y)
        print("-"*60)
        print(f"\n\nThe accuracy of the model is: {accuracy*100} %\n\n")
        print("-"*60)

    def prediction(self):
        predicted_value = self.model.predict([[41, 2, 10, 9, 4, 0, 0, 40]])

        if predicted_value == [0]:
            print("-"*60)
            print("The given person has salary lower than 50K")
            print("-"*60)
        else:
            print("-"*60)
            print("The given person has salary above or equal to 50K")
            print("-"*60)


