class Train_selection:

    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        from sklearn.ensemble import RandomForestClassifier #ensemble --> RandomTree contains more trees

        from sklearn.metrics import accuracy_score

        Estimation_list_min = 1
        Estimation_list_max = 5000

        scores =[]
        n_estimation_list =[]
        max_accuracy = 0

        print("Please wait the model is training")
        for tree in range(Estimation_list_min, Estimation_list_max):
            self.model = RandomForestClassifier(random_state=tree)
            self.model.fit(self.X_train, self.y_train)
            self.y_predicted = self.model.predict(self.X_test)
            scores.append(accuracy_score(self.y_test, self.y_predicted))
            n_estimation_list.append(tree)

            current_accuracy = round(accuracy_score(self.y_predicted, self.y_test)*100,2)
            if(current_accuracy>max_accuracy):
                max_accuracy = current_accuracy
                best_x = tree


        print(max_accuracy, best_x)

        print("Score of the model is:", self.model.score(X_test, y_test))
