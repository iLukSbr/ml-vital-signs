import numpy as np
from .algorithm import Algorithm

class SymbolicC45(Algorithm):
    """
    Implementação detalhada do algoritmo C4.5 para classificação.
    Suporta atributos contínuos, cálculo do ganho de informação normalizado (gain ratio),
    tratamento de valores ausentes e poda pós-processamento (error-based pruning).
    """

    class Node:
        """
        Classe interna que representa um nó da árvore de decisão.

        :param feature_idx: Índice da feature usada para split neste nó.
        :param threshold: Valor de threshold para split (atributos contínuos).
        :param left: Subárvore à esquerda.
        :param right: Subárvore à direita.
        :param value: Classe da folha (se for folha).
        :param is_leaf: Booleano indicando se o nó é folha.
        """
        def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
            self.feature_idx = feature_idx  # Índice da feature usada para split neste nó
            self.threshold = threshold      # Valor de threshold para split (atributos contínuos)
            self.left = left                # Subárvore à esquerda
            self.right = right              # Subárvore à direita
            self.value = value              # Classe da folha (se for folha)
            self.is_leaf = is_leaf          # Booleano indicando se o nó é folha

    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-7, prune=True):
        """
        Inicializa o classificador SymbolicC45.

        :param max_depth: Profundidade máxima da árvore.
        :param min_samples_split: Número mínimo de amostras para tentar split.
        :param min_gain: Ganho mínimo para aceitar um split.
        :param prune: Se True, realiza poda pós-processamento.
        """
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.prune = prune
        self.root = None

    def _entropy(self, y):
        """
        Calcula a entropia do vetor de classes y.

        :param y: Array de classes.
        :return: Valor da entropia.
        """
        # classes: valores únicos em y (as classes possíveis)
        # counts: quantidade de ocorrências de cada classe
        classes, counts = np.unique(y, return_counts=True)
        # probs: probabilidade de cada classe
        probs = counts / counts.sum()
        # Entropia: mede a "impureza" do nó
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _split_info(self, y_left, y_right):
        """
        Calcula a informação do split (split info) para gain ratio.

        :param y_left: Classes do lado esquerdo do split.
        :param y_right: Classes do lado direito do split.
        :return: Valor do split info.
        """
        n = len(y_left) + len(y_right)  # Total de amostras
        p_left = len(y_left) / n if n > 0 else 0  # Proporção à esquerda
        p_right = len(y_right) / n if n > 0 else 0  # Proporção à direita
        split_info = 0
        # Split info penaliza splits muito desbalanceados
        if p_left > 0:
            split_info -= p_left * np.log2(p_left)
        if p_right > 0:
            split_info -= p_right * np.log2(p_right)
        return split_info

    def _information_gain(self, y, y_left, y_right):
        """
        Calcula o ganho de informação de um split.

        :param y: Classes do nó atual.
        :param y_left: Classes do lado esquerdo.
        :param y_right: Classes do lado direito.
        :return: Valor do ganho de informação.
        """
        H = self._entropy(y)  # Entropia do nó atual
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        if n_left == 0 or n_right == 0:
            return 0
        H_left = self._entropy(y_left)    # Entropia do nó esquerdo
        H_right = self._entropy(y_right)  # Entropia do nó direito
        # Ganho de informação: redução da entropia após o split
        gain = H - (n_left / n) * H_left - (n_right / n) * H_right
        return gain

    def _gain_ratio(self, y, y_left, y_right):
        """
        Calcula o gain ratio de um split.

        :param y: Classes do nó atual.
        :param y_left: Classes do lado esquerdo.
        :param y_right: Classes do lado direito.
        :return: Valor do gain ratio.
        """
        gain = self._information_gain(y, y_left, y_right)  # Ganho de informação
        split_info = self._split_info(y_left, y_right)     # Informação do split
        if split_info == 0:
            return 0
        # Gain ratio: normaliza o ganho de informação pelo split info
        return gain / split_info

    def _best_split(self, X, y):
        """
        Encontra o melhor split possível para o nó atual.

        :param X: Matriz de features (shape: [n amostras, n features]).
        :param y: Array de classes.
        :return: Índice da melhor feature, threshold e gain ratio.
        """
        best_gain_ratio = -np.inf  # Melhor gain ratio encontrado
        best_feature = None        # Índice da melhor feature
        best_threshold = None      # Melhor threshold para split
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            values = X[:, feature_idx]
            # Remove NaN para split (ignora valores ausentes)
            mask = ~np.isnan(values)
            if np.sum(mask) < self.min_samples_split:
                continue
            unique_values = np.unique(values[mask])
            if len(unique_values) == 1:
                continue
            # Para atributos contínuos, testamos thresholds entre valores únicos adjacentes
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = values > threshold
                y_left, y_right = y[left_mask], y[right_mask]
                # Split só é válido se ambos os lados tiverem amostras suficientes
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue
                gain_ratio = self._gain_ratio(y, y_left, y_right)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain_ratio

    def _majority_class(self, y):
        """
        Retorna a classe majoritária em y.

        :param y: Array de classes.
        :return: Classe majoritária.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _build_tree(self, X, y, depth):
        """
        Constrói recursivamente a árvore de decisão.

        :param X: Matriz de features.
        :param y: Array de classes.
        :param depth: Profundidade atual.
        :return: Nó da árvore.
        """
        # Critério de parada: todas as amostras são da mesma classe
        if len(np.unique(y)) == 1:
            return self.Node(value=y[0], is_leaf=True)
        # Critério de parada: profundidade máxima atingida ou poucas amostras
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return self.Node(value=self._majority_class(y), is_leaf=True)

        feature_idx, threshold, gain_ratio = self._best_split(X, y)
        # Critério de parada: não há split válido ou ganho insuficiente
        if feature_idx is None or gain_ratio < self.min_gain:
            return self.Node(value=self._majority_class(y), is_leaf=True)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return self.Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        """
        Treina a árvore de decisão C4.5.

        :param X: Matriz de features.
        :param y: Array de classes.
        """
        self.X_train = X
        self.y_train = y
        self.root = self._build_tree(X, y, 0)
        if self.prune:
            self._prune(self.root, X, y)
        self.is_trained = True

    def _prune(self, node, X, y):
        """
        Realiza poda pós-processamento baseada em erro estimado (error-based pruning).

        :param node: Nó atual.
        :param X: Matriz de features.
        :param y: Array de classes.
        """
        if node.is_leaf or X.shape[0] == 0:
            return
        left_mask = X[:, node.feature_idx] <= node.threshold
        right_mask = X[:, node.feature_idx] > node.threshold
        if node.left:
            self._prune(node.left, X[left_mask], y[left_mask])
        if node.right:
            self._prune(node.right, X[right_mask], y[right_mask])
        # Se ambos os filhos são folhas, verifica se pode podar
        if node.left and node.right and node.left.is_leaf and node.right.is_leaf:
            left_errors = np.sum(y[left_mask] != node.left.value)   # Erros no filho esquerdo
            right_errors = np.sum(y[right_mask] != node.right.value) # Erros no filho direito
            total_errors = left_errors + right_errors                # Erros totais dos filhos
            leaf_errors = np.sum(y != self._majority_class(y))       # Erros se virar folha
            # Se a folha teria menos erro, poda
            if leaf_errors <= total_errors:
                node.value = self._majority_class(y)
                node.left = None
                node.right = None
                node.is_leaf = True

    def _predict_one(self, x, node):
        """
        Realiza a predição de uma única amostra percorrendo a árvore.

        :param x: Vetor de features da amostra.
        :param node: Nó atual da árvore.
        :return: Classe prevista.
        """
        if node.is_leaf:
            return node.value
        if np.isnan(x[node.feature_idx]):
            # Valor ausente: retorna a classe majoritária do nó
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        """
        Realiza a predição para um conjunto de amostras.

        :param X: Matriz de features.
        :return: Array de classes previstas.
        """
        return np.array([self._predict_one(x, self.root) for x in X])
