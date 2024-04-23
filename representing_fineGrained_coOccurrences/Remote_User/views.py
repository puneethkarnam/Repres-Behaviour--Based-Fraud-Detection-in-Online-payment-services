from django.shortcuts import render, redirect, get_object_or_404
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix, classification_report

from Remote_User.models import ClientRegister_Model,fraud_detection_type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Financial_Fraud_detection_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            step= request.POST.get('step')
            Product_name= request.POST.get('Product_name')
            type= request.POST.get('type')
            amount= request.POST.get('amount')
            nameOrig= request.POST.get('nameOrig')
            oldbalanceOrg= request.POST.get('oldbalanceOrg')
            newbalanceOrig= request.POST.get('newbalanceOrig')
            nameDest= request.POST.get('nameDest')
            oldbalanceDest= request.POST.get('oldbalanceDest')
            newbalanceDest= request.POST.get('newbalanceDest')

        data = pd.read_csv("Online_Payment_Datasets.csv", encoding='latin-1')
        # data.replace([np.inf, -np.inf], np.nan, inplace=True)

        data['Results'] = data.isFraud.apply(lambda x: 1 if x == 1 else 0)

        x = data['nameOrig']
        y = data['Results']

        # data.drop(['Type_of_Breach'],axis = 1, inplace = True)
        cv = CountVectorizer()

        print(x)
        print(y)

        x = cv.fit_transform(x)


        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        nameOrig1 = [nameOrig]
        vector1 = cv.transform(nameOrig1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'No Fraud'
        elif prediction == 1:
            val = 'Fraud'

        fraud_detection_type.objects.create(step=step,
        Product_name=Product_name,
        type=type,
        amount=amount,
        nameOrig=nameOrig,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        nameDest=nameDest,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
        Prediction=val)

        return render(request, 'RUser/Predict_Financial_Fraud_detection_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Financial_Fraud_detection_Type.html')



