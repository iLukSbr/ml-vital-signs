import numpy as np
from scipy.stats import mode
from sklearn.utils import resample
from .algorithm import Algorithm

class SymbolicNode:
    """
    @class SymbolicNode
    @brief Nó de uma árvore simbólica. Pode ser uma operação matemática ou uma folha (variável ou constante).

    @var op Operação do nó ('+', '-', '*', '/', ou None para folha).
    @var left Filho esquerdo (SymbolicNode).
    @var right Filho direito (SymbolicNode).
    @var value Valor constante (se folha constante).
    @var feature_idx Índice da feature (se folha variável).
    """
    def __init__(self, op=None, left=None, right=None, value=None, feature_idx=None):
        """
        @brief Inicializa um nó simbólico.
        @param op Operação do nó ('+', '-', '*', '/', ou None).
        @param left Filho esquerdo.
        @param right Filho direito.
        @param value Valor constante (se folha).
        @param feature_idx Índice da feature (se folha variável).
        """
        self.op = op
        self.left = left
        self.right = right
        self.value = value
        self.feature_idx = feature_idx

    def evaluate(self, X):
        """
        @brief Avalia o nó simbólico para um conjunto de amostras.
        @param X np.ndarray de features (shape: [n amostras, n features]).
        @return np.ndarray com o resultado da avaliação para cada amostra.
        """
        if self.op is None:
            if self.feature_idx is not None:
                # Folha variável: retorna a coluna correspondente de X
                return X[:, self.feature_idx]
            else:
                # Folha constante: retorna um array constante
                return np.full(X.shape[0], self.value)
        else:
            # Nó interno: avalia filhos e aplica operação
            left_val = self.left.evaluate(X)
            right_val = self.right.evaluate(X)
            if self.op == '+':
                return left_val + right_val  # soma elemento a elemento
            elif self.op == '-':
                return left_val - right_val  # subtração elemento a elemento
            elif self.op == '*':
                return left_val * right_val  # multiplicação elemento a elemento
            elif self.op == '/':
                # Evita divisão por zero usando np.where
                return np.where(np.abs(right_val) > 1e-8, left_val / right_val, 0.0)
            else:
                raise ValueError(f"Operação desconhecida: {self.op}")

class SymbolicTree:
    """
    @class SymbolicTree
    @brief Árvore simbólica simples construída aleatoriamente.

    @var max_depth Profundidade máxima da árvore.
    @var n_features Número de features de entrada.
    @var root Raiz da árvore (SymbolicNode).
    @var random_state Gerador de números aleatórios.
    """
    def __init__(self, max_depth=3, n_features=None, random_state=None):
        """
        @brief Inicializa a árvore simbólica.
        @param max_depth Profundidade máxima da árvore.
        @param n_features Número de features de entrada.
        @param random_state Semente para reprodutibilidade.
        """
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.random_state = np.random.RandomState(random_state)

    def _random_node(self, depth):
        """
        @brief Cria recursivamente um nó simbólico aleatório.
        @param depth Profundidade atual.
        @return SymbolicNode criado.
        """
        if depth >= self.max_depth or self.random_state.rand() < 0.3:
            # Nó folha: variável ou constante
            if self.random_state.rand() < 0.5:
                # Constante aleatória entre -10 e 10
                value = self.random_state.uniform(-10, 10)
                return SymbolicNode(value=value)
            else:
                # Variável aleatória (feature)
                feature_idx = self.random_state.randint(0, self.n_features)
                return SymbolicNode(feature_idx=feature_idx)
        else:
            # Nó interno: operação binária aleatória
            op = self.random_state.choice(['+', '-', '*', '/'])
            left = self._random_node(depth + 1)
            right = self._random_node(depth + 1)
            return SymbolicNode(op=op, left=left, right=right)

    def fit(self, X, y):
        """
        @brief Constrói a árvore simbólica aleatória.
        @param X Dados de entrada (features).
        @param y Rótulos de classe (não usado nesta implementação).
        """
        self.n_features = X.shape[1]
        self.root = self._random_node(0)
        # Opcional: ajuste de constantes pode ser feito aqui (ex: otimização simbólica)

    def predict(self, X):
        """
        @brief Realiza a predição para um conjunto de amostras.
        @param X Dados de entrada (features).
        @return np.ndarray com as predições.
        """
        return self.root.evaluate(X)

class SymbolicRandomForest(Algorithm):
    """
    @class SymbolicRandomForest
    @brief Random Forest Simbólico baseado na classe Algorithm.

    @var n_estimators int - Número de árvores na floresta.
    @var max_depth int - Profundidade máxima de cada árvore simbólica.
    @var random_state int|None - Semente para reprodutibilidade dos resultados.
    @var bootstrap bool - Se True, utiliza amostragem com reposição (bootstrap) para cada árvore.
    @var trees list - Lista de instâncias de SymbolicTree treinadas.
    @var classes_ np.ndarray - Array com os rótulos de classe únicos vistos no treino.
    """

    def __init__(self, n_estimators=10, max_depth=3, random_state=None, bootstrap=True):
        """
        @brief Inicializa o Random Forest Simbólico.
        @param n_estimators int - Número de árvores na floresta.
        @param max_depth int - Profundidade máxima de cada árvore.
        @param random_state int|None - Semente para reprodutibilidade.
        @param bootstrap bool - Se True, utiliza amostragem com reposição.
        """
        super().__init__()
        self.n_estimators = n_estimators  # @var n_estimators
        self.max_depth = max_depth        # @var max_depth
        self.random_state = random_state  # @var random_state
        self.bootstrap = bootstrap        # @var bootstrap
        self.trees = []                  # @var trees
        self.classes_ = None             # @var classes_

    def fit(self, X, y):
        """
        @brief Treina o Random Forest Simbólico.
        @param X np.ndarray - Dados de entrada (amostras x features).
        @param y np.ndarray - Rótulos de classe.
        """
        self.X_train = X  # @var X_train
        self.y_train = y  # @var y_train
        self.trees = []   # Limpa árvores anteriores
        self.classes_ = np.unique(y)  # @var classes_ - Salva as classes vistas no treino
        rng = np.random.RandomState(self.random_state)  # Gerador de números aleatórios

        for i in range(self.n_estimators):
            # @var i int - Índice da árvore atual
            if self.bootstrap:
                # Amostragem com reposição (bootstrap)
                X_sample, y_sample = resample(X, y, random_state=rng.randint(0, 10000))
            else:
                # Usa todos os dados sem bootstrap
                X_sample, y_sample = X, y
            # Cria e treina uma árvore simbólica
            tree = SymbolicTree(max_depth=self.max_depth, random_state=rng.randint(0, 10000))
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)  # Adiciona à floresta
        self.is_trained = True  # Marca como treinado

    def predict(self, X):
        """
        @brief Realiza a predição agregada das árvores simbólicas.
        @param X np.ndarray - Dados de entrada (amostras x features).
        @return np.ndarray - Rótulos de classe previstos (apenas valores vistos no treino).
        """
        # Coleta as predições contínuas de todas as árvores
        preds = np.array([tree.predict(X) for tree in self.trees])  # shape: (n_trees, n_amostras)
        # @var preds np.ndarray - Previsões contínuas de cada árvore

        # Para cada árvore, converte cada predição para o rótulo de classe mais próximo
        preds_classes = np.empty_like(preds, dtype=self.classes_.dtype)  # @var preds_classes np.ndarray
        for i, c in enumerate(self.classes_):
            # @var i int - Índice da classe
            # @var c valor da classe
            if i == 0:
                min_dist = np.abs(preds - c)  # @var min_dist np.ndarray - Distância inicial
                preds_classes[:] = c
            else:
                closer = np.abs(preds - c) < min_dist  # @var closer np.ndarray - Máscara de proximidade
                preds_classes[closer] = c
                min_dist = np.where(closer, np.abs(preds - c), min_dist)
        # Votação majoritária entre as árvores para cada amostra
        from scipy.stats import mode
        majority_vote = mode(preds_classes, axis=0, keepdims=False).mode  # @var majority_vote np.ndarray
        return majority_vote.flatten()
