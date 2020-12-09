import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets.samples_generator import make_blobs
from IPython.display import display
from nltk import word_tokenize
from nltk.corpus import stopwords

implementations = ['Simple Linear Regression', 'Naive Bayes', 'Random Forest', 'Support Vector Machines']

def run(algo):
    assert algo in implementations

    if algo == 'Simple Linear Regression':
        X, y = make_regression(n_samples=100, n_features=1, noise=1, random_state=18)
        print("Building Modeling...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        print("Fitting model to data...")
        regressor = LinearRegression().fit(X_train, y_train)
        print("Generating Predictions...")
        y_pred = regressor.predict(X_test)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(X_train, y_train, color='gray', label='training data')
        ax.scatter(X_test, y_test, color='red', label='test data')
        ax.plot(X_test, y_pred, color='dodgerblue', linewidth=2, label='line of best fit')
        ax.set(xlabel='feature 1', ylabel='target', title='Simple Linear Regression Implementation')
        _ = ax.legend()

        print()

        df = pd.DataFrame({'prediction': y_pred, 'actual': y_test})
        df.iloc[:15,:].plot(kind='bar', figsize=(8,6))
        _ = plt.title('Predictions')

    
    elif algo == 'Naive Bayes':
        print("Preview of Amazon Reviews Dataset (Uncleaned): ")
        amazon_reviews = pd.read_csv("datasets/amazon_cells_labelled.csv", header=None)
        display(amazon_reviews.head(10))
        
        print("Cleaning Dataset...")
        print()
        
        X = []
        y = []

        for i, record in enumerate(amazon_reviews.iterrows()):
            for j in range(len(record[1])):
                if (pd.isnull(record[1][j]) == True):
                    sentiment = record[1][j-1]
                    reviews = record[1][:j-1]
                    break
                elif j == (len(record[1]) - 1):
                    sentiment = record[1][j]
                    reviews = record[1][:j]

            reviews = reviews.str.cat(sep='')
            X.append(reviews)
            y.append(sentiment)

        ar = pd.DataFrame({'review': X, 'sentiment': y}, index=np.arange(len(amazon_reviews.index)))
        ar["review"] = ar['review'].str.replace('[^\w\s]','').str.lower()
        ar.sentiment = ar.sentiment.astype(int)

        print("Preview of Amazon Reviews Dataset (After Data Cleaning Phase): ")
        display(ar.head(10))
        print()
        
        print("Data Preprocessing...")
        
        stop = set(stopwords.words('english'))

        reviews = ar.review
        y = ar.sentiment

        # split reviews into terms
        review_words = [review.split() for review in reviews]


        vocab = sorted(set(sum(review_words, [])))
        vocab_dict = {k:i for i,k in enumerate(vocab)}

        X_tf = np.zeros((len(reviews), len(vocab)), dtype=int)

        for i, review in enumerate(review_words):
            for word in review:
                X_tf[i, vocab_dict[word]] += 1

        X = X_tf

        review_words = [review.split() for review in reviews]
        vocab = sorted(set(sum(review_words, [])))
        vocab_wo_stop_words = [word for word in vocab if word not in stop]

        print(f"Removed {len(vocab) - len(vocab_wo_stop_words)} stop words.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
    
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        print()
        print(f"Multinomial Naive Bayes Accuracy: {clf.score(X,y)}")
        
        train_acc = []
        test_acc = []

        for i in range(10, 660, 50):
            reviews = ar.review[:i]
            y = ar.sentiment[:i]

            review_words = [review.split() for review in reviews]


            vocab = sorted(set(sum(review_words, [])))
            vocab_dict = {k:i for i,k in enumerate(vocab)}

            X_tf = np.zeros((len(reviews), len(vocab)), dtype=int)
            for i, review in enumerate(review_words):
                for word in review:
                    X_tf[i, vocab_dict[word]] += 1

            X = X_tf

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

            clf = MultinomialNB()
            clf.fit(X_train, y_train)


            #print(clf.score(X_train, y_train))
            train_acc.append(clf.score(X_train, y_train))
            #print(clf.score(X_test, y_test))
            test_acc.append(clf.score(X_test, y_test))
    
        s = np.arange(10, 660, 50)

        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(s,train_acc, c='blue', label='training accuracy')
        ax.plot(s,test_acc, c='red', label='test accuracy')
        ax.set(xlabel='number of samples', ylabel='accuracy', title='Dataset Size and Generalization Performance')
        _ = ax.legend()
    
    elif algo == 'Random Forest':
        X, y = make_moons(n_samples=100, noise=0.25, random_state=12)
        cmap2 = ListedColormap(['tab:red', 'tab:blue'])

        def scatter_moons(X, y, cmap=cmap2, ax=None, size=50):
            if ax is None:
                fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, linewidth=1, s=size)
            ax.set(title="Toy 2D Moon Dataset", xlabel="Feature 0", ylabel="Feature1")

        scatter_moons(X,y, cmap=cmap2)

        def plot_decision_boundary(clf, X, y, cmap, ax=None, title=None, alpha=0.25, size=60):
            if ax is None:
                fig, ax = plt.subplots(figsize=(8,7))
            # scatter the data
            scatter_moons(X,y, ax=ax, cmap=cmap, size=size)
            h = 0.01 # step size in the mesh
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x_array = np.arange(xmin, xmax, h)
            y_array = np.arange(ymin, ymax, h)
            xx, yy = np.meshgrid(x_array, y_array)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cmap, alpha=alpha)
            ax.contour(xx, yy, Z, colors='k', linewidths=1)
            if title != None:
                ax.set_title("{}".format(title))

        forest = RandomForestClassifier(n_estimators=6, random_state=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        forest.fit(X_train, y_train)

        print("Accuracy on training set: {:.2f}".format(forest.score(X_train, y_train)))
        print("Accuracy on test set: {:.2f}".format(forest.score(X_test, y_test)))

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        for i, (ax, tree) in enumerate(zip(axes.flat, forest.estimators_)):
            plot_decision_boundary(tree,X,y,cmap=cmap2, ax=ax, title="Tree {}".format(i))

        fig, ax = plt.subplots(figsize=(8,6))
        plot_decision_boundary(forest,X,y,cmap=cmap2, ax=ax, title="Random Forest")
    
    elif algo == 'Support Vector Machines':
        X, y = make_circles(n_samples=300, noise=0.06, factor=0.5, random_state=0)
        cmap1 = ListedColormap(['tab:red', 'tab:blue'])

        def scatter_circles(X, y, cmap=cmap1, ax=None, title=None):
            if ax is None:
                fig, ax = plt.subplots(figsize=(8,7))
            ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, s=40)
            ax.set(title="Toy 2D Circles Dataset", xlabel="Feature 0", ylabel="Feature1")
            if title != None:
                ax.set(title=title)

        target = y
        x = X[:,0]
        y = X[:,1]
        z = x**2 + y**2

        fig, ax = plt.subplots(figsize=(8,6))
        scatter_circles(X, target, ax=ax, title="Original Feature Representation")

        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(y,z,c=target, cmap=cmap1)
        features = np.c_[y,z] 
        svc = LinearSVC().fit(features, target)
        coef, intercept = svc.coef_, svc.intercept_
        line = np.linspace(-2,2)
        for coef, intercept in zip(svc.coef_, svc.intercept_):
            ax.plot(line, (-(line * coef[0] + intercept) / coef[1]), c='k') 
        _ = ax.set(title="Expanded Feature Representation on the Y-Z Plane", xlabel="Feature 1 (Y)", ylabel="Feature 2 (Z)")

        X, y = make_blobs(n_samples=50, centers=2, n_features=2, cluster_std=3.5, random_state=9)
        cmap2 = ListedColormap(['deepskyblue', 'gold'])

        def plot_blobs(X, y, cmap=cmap2, ax=None):
            if ax is None:
                fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap)
            ax.set_xlim(np.round(X[:,0].min() - 1 ), np.round(X[:,0].max() + 1))
            ax.set_ylim(np.round(X[:,1].min() - 1 ), np.round(X[:,1].max() + 1))
            ax.set(title="Toy 2D Dataset", xlabel="Feature 0", ylabel="Feature1")

        def plot_decision_boundary(clf, X, y, cmap=cmap2, ax=None, title=None):
            if ax is None:
                fig, ax = plt.subplots(figsize=(8,7))

            # scatter data
            plot_blobs(X,y, ax=ax, cmap=cmap)

            h = 0.01 # step size in the mesh
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x_array = np.arange(xmin, xmax, h)
            y_array = np.arange(ymin, ymax, h)
            xx, yy = np.meshgrid(x_array, y_array)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
            if title != None:
                ax.set_title("{}".format(title))

        C = 1.0  # SVM regularization parameter
        svc = SVC(kernel='linear', C=C).fit(X, y)
        lin_svc = LinearSVC(C=C).fit(X, y)
        rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
        poly_svc = SVC(kernel='poly', degree=3, C=C).fit(X, y)


        titles = ["Decision Boundary for SVC with Linear Kernel", 
                  "Decision Boundary for LinearSVC (linear kernel)", 
                  "Decision Boundary for SVC with RBF (Gaussian) Kernel", 
                  "Decision Boundary for SVC with Polynomial Kernel",]
        clfs =[svc, lin_svc, rbf_svc, poly_svc]

        fig, axes = plt.subplots(2,2,figsize=(15,10))

        for ax, title, clf in zip(axes.flat, titles, clfs):
            plot_decision_boundary(clf, X, y, cmap=cmap2, ax=ax, title=title)
    
    else:
        return




