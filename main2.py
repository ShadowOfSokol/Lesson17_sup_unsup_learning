'''
Проєкт "Аналіз відвідувачів кафе"
Опис проєкту:
Цей проєкт досліджує взаємозв’язок між кількістю відвідувачів кафе
та доходом, а також дозволяє сегментувати відвідувачів за їхньою поведінкою
(частота візитів і середня сума замовлення).
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def main():
    # Data generation
    np.random.seed(42)
    visitors = np.random.randint(6,45, 30)
    revenue = 10 * visitors + np.random.randint(600,9000, 30)
    data = {
        "Visitors": visitors,
        "Revenue": revenue,
    }
    df = pd.DataFrame(data)

    # Data divide
    X = df[["Visitors"]]
    Y = df["Revenue"]

    # Model
    model = LinearRegression()
    model.fit(X, Y)
    # Prediction
    Y_pred = model.predict(X)

    # Visualization
    plt.scatter(X, Y, color="blue", label = "Data")
    plt.plot(X, Y_pred, color="red", label="Prediction")
    plt.title("Finale Revenue Amount")
    plt.xlabel("Visitors")
    plt.ylabel("Revenue")
    plt.legend()
    plt.show()

    # Prediction


    # Clusterization

    # Data divide

    # Model

    # Prediction

if __name__ == '__main__':
    main()