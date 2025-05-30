import numpy as np
from abc import ABC, abstractmethod

class Algorithm(ABC):
    """
    @class Algorithm
    @brief Classe base abstrata para todos os modelos de Aprendizado de Máquina.
    Define funções e variáveis comuns para Random Forest Simbólico, C4.5 e Redes Neurais.
    """

    def __init__(self):
        """
        @brief Inicializa as variáveis comuns do modelo.
        @var model Instância do modelo (pode ser sobrescrita pelas subclasses).
        @var is_trained Booleano indicando se o modelo já foi treinado.
        @var X_train Dados de treino (features).
        @var y_train Rótulos de treino.
        """
        self.model = None
        self.is_trained = False
        self.X_train = None
        self.y_train = None

    @abstractmethod
    def fit(self, X, y):
        """
        @brief Treina o modelo com os dados de entrada X e rótulos y.
        @param X Dados de entrada (features), array-like ou np.ndarray.
        @param y Rótulos de classe, array-like ou np.ndarray.
        """
        self.X_train = X  # armazena os dados de treino
        self.y_train = y  # armazena os rótulos de treino
        self.is_trained = True  # marca como treinado

    @abstractmethod
    def predict(self, X):
        """
        @brief Realiza previsões com base nos dados de entrada X.
        @param X Dados de entrada (features), array-like ou np.ndarray.
        @return np.ndarray com as previsões.
        """
        pass

    def score(self, X, y, metric):
        """
        @brief Avalia o modelo usando uma métrica fornecida.
        @param X Dados de entrada (features) para avaliação.
        @param y Rótulos verdadeiros.
        @param metric Função de métrica (ex: accuracy, f1, etc).
        @return Resultado da métrica aplicada às previsões.
        """
        y_pred = self.predict(X)  # obtém as previsões do modelo
        return metric(y, y_pred)  # calcula e retorna a métrica

    def is_fitted(self):
        """
        @brief Verifica se o modelo já foi treinado.
        @return True se treinado, False caso contrário.
        """
        return self.is_trained
