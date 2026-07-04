import matplotlib.pyplot as plt
import mglearn
from imageio import imread
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import imageio



# adapted from mglearn https://github.com/amueller/mglearn/blob/master/mglearn/tools.py
def my_heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    
    iteration = 0
    for p, color, value in zip(img.get_paths(), img.get_facecolors(),
                               img.get_array().flatten()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'        
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img

from scipy.stats import norm
def plot_gaussians(df, feat_names=['Weight (in grams)', 'Sugar Content (in %)']):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Iterate over each feature
    for ax, feature in zip(axes, feat_names):
        for fruit, color in [('Apple', 'red'), ('Orange', 'orange')]:
            subset = df[df['Fruit'] == fruit]
            mu = subset[feature].mean()
            sigma = subset[feature].std()
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
            ax.plot(x, norm.pdf(x, mu, sigma), c=color, label=f'{fruit} Gaussian')
            ax.set_title(f'Gaussian for {feature}')
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    
def plot_top_k_words(data, k=20, target="spam"):
    countvec = CountVectorizer(binary=True, stop_words='english')
    data_vec = countvec.fit_transform(data)
    feature_names = countvec.get_feature_names_out()    
    word_counts = data_vec.sum(axis=0).A1
    sorted_indices = np.argsort(word_counts)[::-1][:k]
    top_k_words = [feature_names[i] for i in sorted_indices]
    top_k_counts = word_counts[sorted_indices]
    
    # plot
    plt.figure(figsize=(8, 6))
    plt.bar(top_k_words, top_k_counts)
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Frequencies')
    plt.title(f'Top k Words and Frequencies for target: {target}')
    plt.show()

def plot_word_freq(data, query_words=['block', 'free', 'prize', 'urgent'], target="target"):
    countvec = CountVectorizer(binary=True, stop_words='english')
    data_vec = countvec.fit_transform(data)
    feature_names = countvec.get_feature_names_out()    
    word_counts = data_vec.sum(axis=0).A1
    word_count_dict = dict(zip(feature_names, word_counts))
    query_counts = {word: word_count_dict.get(word, 0) for word in query_words}
    return query_counts

def plot_spam_ham_word_freq(spam_data, ham_data):
    #plot
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 4))
    query_counts = plot_word_freq(spam_data, target="spam")
    spam_words = list(query_counts.keys())
    spam_counts = list(query_counts.values())
    ax[0].bar(spam_words, spam_counts)   
    ax[0].set_xticklabels(labels=spam_words, rotation=45)    
    ax[0].set_xlabel('Words')
    ax[0].set_ylabel('Frequencies')
    ax[0].set_title(f'spam words frequencies')
    
    query_counts = plot_word_freq(ham_data, target="ham")
    ham_words = list(query_counts.keys())
    ham_counts = list(query_counts.values())
    ax[1].bar(ham_words, ham_counts)   
    ax[1].set_xticklabels(labels=ham_words, rotation=45)    
    ax[1].set_xlabel('Words')    
    ax[1].set_title(f'ham words frequencies')
    

def plot_tree_decision_boundary(
    model, X, y, x_label="x-axis", y_label="y-axis", eps=None, ax=None, title=None
):
    if ax is None:
        ax = plt.gca()

    if title is None:
        title = "max_depth=%d" % (model.tree_.max_depth)

    mglearn.plots.plot_2d_separator(
        model, X.to_numpy(), eps=eps, fill=True, alpha=0.5, ax=ax
    )
    mglearn.discrete_scatter(X.iloc[:, 0], X.iloc[:, 1], y, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_tree_decision_boundary_and_tree(
    model, X, y, height=6, width=16, fontsize = 9, x_label="x-axis", y_label="y-axis", eps=None
):
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(width, height),
        subplot_kw={"xticks": (), "yticks": ()},
        gridspec_kw={"width_ratios": [1.5, 2]},
    )
    plot_tree_decision_boundary(model, X, y, x_label, y_label, eps, ax=ax[0])
    custom_plot_tree(model, 
                 feature_names=X.columns.tolist(), 
                 class_names=['A+', 'not A+'],
                 impurity=False,
                 fontsize=fontsize, ax=ax[1])
    ax[1].set_axis_off()
    plt.show()

def plot_knn_decision_boundaries(X_train, y_train, k_values = [1,11,100]):
    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 4))

    for n_neighbors, ax in zip(k_values, axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_validate(clf, X_train, y_train, return_train_score=True)
        mean_valid_score = scores["test_score"].mean()
        mean_train_score = scores["train_score"].mean()
        clf.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(
            clf, X_train.to_numpy(), fill=True, eps=0.5, ax=ax, alpha=0.4
        )
        mglearn.discrete_scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], y_train, ax=ax)
        title = "n_neighbors={}\n train score={}, valid score={}".format(
            n_neighbors, round(mean_train_score, 2), round(mean_valid_score, 2)
        )
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    axes[0].legend(loc=1);    

def plot_train_test_points(X_train, y_train, X_test, class_names=['class 0','class 1'], test_format='star'):
    training_points = mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    if test_format == "circle": 
        test_points = mglearn.discrete_scatter(
                X_test[:, 0], X_test[:, 1], markers="o", c='k', s=18
            );
    else: 
        test_points = mglearn.discrete_scatter(
                X_test[:, 0], X_test[:, 1], markers="*", c='g', s=16
            );        
    plt.legend(
        training_points + test_points,
        [class_names[0], class_names[1], "test point(s)"],
    )  
