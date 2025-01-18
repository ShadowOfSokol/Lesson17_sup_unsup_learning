'''
Проєкт "Аналіз спортивної команди"
Опис проєкту:
Проєкт допомагає зрозуміти, як тренування впливають
на результати гри команди, а також розподілити гравців
за їхніми здібностями, щоб оптимізувати тренувальний процес.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def main():
    # Data generation
    np.random.seed(42)
    training_hours = np.random.randint(0,8, 30)
    game_points = 10 * training_hours + np.random.randint(0,50, 30)
    data = {
        "Training Hours": training_hours,
        "Game Points": game_points,
    }
    df = pd.DataFrame(data)

    # Data divide
    X = df[["Training Hours"]]
    Y = df["Game Points"]

    # Model
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    # Visualization
    plt.scatter(X, Y, color="blue", label = "Data")
    plt.plot(X, Y_pred, color="red", label="Prediction")
    plt.title("Finale Points Amount")
    plt.xlabel("Training Hours")
    plt.ylabel("Game Points")
    plt.legend()
    plt.show()

    # Prediction


    # Clusterization

    # Data divide

    # Model

    # Prediction

if __name__ == '__main__':
    main()