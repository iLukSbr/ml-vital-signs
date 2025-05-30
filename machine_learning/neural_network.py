import numpy as np
from .algorithm import Algorithm

class NeuralNetwork(Algorithm):
    """
    @class NeuralNetwork
    @brief Implementação manual de um Perceptron Multicamadas (MLP) em numpy, com avanços recentes.
    Suporta múltiplas camadas ocultas, funções de ativação customizáveis (incluindo Swish, GELU, Mish),
    dropout, batch normalization, layer normalization, early stopping e pode ser usada para classificação ou regressão.

    @var input_dim: int - Número de neurônios na camada de entrada (features).
    @var output_dim: int - Número de neurônios na camada de saída (classes ou valor contínuo).
    @var hidden_layers: list - Lista com o número de neurônios em cada camada oculta.
    @var activation: str - Função de ativação ('relu', 'tanh', 'sigmoid', 'swish', 'gelu', 'mish', 'leaky_relu', 'elu').
    @var dropout: float - Taxa de dropout (entre 0 e 1).
    @var batch_norm: bool - Se True, aplica Batch Normalization.
    @var layer_norm: bool - Se True, aplica Layer Normalization.
    @var lr: float - Taxa de aprendizado.
    @var batch_size: int - Tamanho do batch para SGD.
    @var epochs: int - Número máximo de épocas de treinamento.
    @var patience: int - Número de épocas sem melhora para early stopping.
    @var task: str - 'classification' ou 'regression'.
    @var verbose: bool - Exibe logs de treinamento.
    @var is_trained: bool - Indica se o modelo já foi treinado.
    @var weights: list - Lista de matrizes de pesos de cada camada.
    @var biases: list - Lista de vetores de bias de cada camada.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers=[128, 64],
        activation='relu',
        dropout=0.0,
        batch_norm=False,
        layer_norm=False,
        lr=1e-3,
        batch_size=64,
        epochs=100,
        patience=10,
        task='classification',
        verbose=True
    ):
        """
        @brief Inicializa o MLP e seus hiperparâmetros.
        @param input_dim: int - Número de features de entrada.
        @param output_dim: int - Número de saídas (classes ou valor de regressão).
        @param hidden_layers: list - Lista com o número de neurônios em cada camada oculta.
        @param activation: str - Função de ativação ('relu', 'tanh', 'sigmoid', 'swish', 'gelu', 'mish', 'leaky_relu', 'elu').
        @param dropout: float - Taxa de dropout (entre 0 e 1).
        @param batch_norm: bool - Se True, aplica Batch Normalization.
        @param layer_norm: bool - Se True, aplica Layer Normalization.
        @param lr: float - Taxa de aprendizado.
        @param batch_size: int - Tamanho do batch.
        @param epochs: int - Número máximo de épocas.
        @param patience: int - Épocas sem melhora para early stopping.
        @param task: str - 'classification' ou 'regression'.
        @param verbose: bool - Exibe logs de treinamento.
        """
        super().__init__()
        self.input_dim = input_dim  # @var input_dim
        self.output_dim = output_dim  # @var output_dim
        self.hidden_layers = hidden_layers  # @var hidden_layers
        self.activation = activation  # @var activation
        self.dropout = dropout  # @var dropout
        self.batch_norm = batch_norm  # @var batch_norm
        self.layer_norm = layer_norm  # @var layer_norm
        self.lr = lr  # @var lr
        self.batch_size = batch_size  # @var batch_size
        self.epochs = epochs  # @var epochs
        self.patience = patience  # @var patience
        self.task = task  # @var task
        self.verbose = verbose  # @var verbose
        self.is_trained = False  # @var is_trained
        self._init_weights()
        if self.batch_norm:
            self._init_batch_norm_params()

    # --- Funções de ativação e derivadas ---
    def _relu(self, x):
        """
        @brief Função de ativação ReLU.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Saída após ReLU.
        """
        return np.maximum(0, x) # ReLU: max(0, x)

    def _relu_deriv(self, x):
        """
        @brief Derivada da ReLU.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Derivada.
        """
        return (x > 0).astype(float) # Derivada: 1 se x > 0, 0 se x <= 0

    def _leaky_relu(self, x, alpha=0.01):
        """
        @brief Função de ativação Leaky ReLU.
        @param x: np.ndarray - Entrada.
        @param alpha: float - Slope para x < 0.
        @return np.ndarray - Saída após Leaky ReLU.
        """
        return np.where(x > 0, x, alpha * x) # Leaky ReLU: x se x > 0, alpha * x se x <= 0

    def _leaky_relu_deriv(self, x, alpha=0.01):
        """
        @brief Derivada da Leaky ReLU.
        @param x: np.ndarray - Entrada.
        @param alpha: float - Slope para x < 0.
        @return np.ndarray - Derivada.
        """
        return np.where(x > 0, 1, alpha) # Derivada: 1 se x > 0, alpha se x <= 0

    def _elu(self, x, alpha=1.0):
            """
            @brief Função de ativação ELU.
            @param x: np.ndarray - Entrada.
            @param alpha: float - Slope para x < 0.
            @return np.ndarray - Saída após ELU.
            """
            x_clip = np.clip(x, -500, 500) # Evita overflow
            return np.where(x > 0, x, alpha * (np.exp(x_clip) - 1)) # ELU: x se x > 0, alpha * (exp(x) - 1) se x <= 0

    def _elu_deriv(self, x, alpha=1.0):
        """
        @brief Derivada da ELU.
        @param x: np.ndarray - Entrada.
        @param alpha: float - Slope para x < 0.
        @return np.ndarray - Derivada.
        """
        x_clip = np.clip(x, -500, 500) # Evita overflow
        return np.where(x > 0, 1, alpha * np.exp(x_clip)) # Derivada: 1 se x > 0, alpha * exp(x) se x <= 0

    def _sigmoid(self, x):
        """
        @brief Função sigmoide.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Saída após sigmoid.
        """
        x = np.clip(x, -500, 500)  # Limita o expoente para evitar overflow
        return 1 / (1 + np.exp(-x)) # Sigmoid: 1 / (1 + exp(-x))

    def _sigmoid_deriv(self, x):
        """
        @brief Derivada da sigmoide.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Derivada.
        """
        s = self._sigmoid(x) # Sigmoid: 1 / (1 + exp(-x))
        return s * (1 - s) # Derivada: s * (1 - s), onde s é a saída da sigmoide

    def _tanh(self, x):
        """
        @brief Função tanh.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Saída após tanh.
        """
        return np.tanh(x) # Tanh: (e^x - e^-x) / (e^x + e^-x)

    def _tanh_deriv(self, x):
        """
        @brief Derivada da tanh.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Derivada.
        """
        return 1 - np.tanh(x) ** 2 # Derivada: 1 - tanh(x)^2

    def _swish(self, x):
        """
        @brief Função de ativação Swish.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Saída após Swish.
        """
        x_clip = np.clip(x, -500, 500) # Evita overflow
        return x * self._sigmoid(x_clip) # Swish: x * sigmoid(x)

    def _swish_deriv(self, x):
        """
        @brief Derivada da Swish.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Derivada.
        """
        s = self._sigmoid(x) # Sigmoid de x
        return s + x * s * (1 - s) # Derivada da Swish: s + x * s * (1 - s)

    def _gelu(self, x):
        """
        @brief Função de ativação GELU.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Saída após GELU.
        """
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3))) # GELU: x * P(X <= x) onde P é a distribuição normal

    def _gelu_deriv(self, x):
        """
        @brief Derivada da GELU.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Derivada.
        """
        tanh_out = np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)) # Tanh de GELU
        return 0.5 * (1 + tanh_out) + 0.5 * x * (1 - tanh_out**2) * (np.sqrt(2/np.pi)*(1 + 3*0.044715*x**2)) # Derivada da GELU: 0.5 * (1 + tanh_out) + 0.5 * x * (1 - tanh_out**2) * (sqrt(2/pi)*(1 + 3*0.044715*x**2))

    def _mish(self, x):
        """
        @brief Função de ativação Mish.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Saída após Mish.
        """
        x_clip = np.clip(x, -500, 500) # Evita overflow
        return x * np.tanh(np.log1p(np.exp(x_clip))) # Mish: x * tanh(log(1 + exp(x)))

    def _mish_deriv(self, x):
        """
        @brief Derivada da Mish.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Derivada.
        """
        x_clip = np.clip(x, -500, 500) # Evita overflow
        sp = self._sigmoid(x_clip) # Sigmoid de x_clip
        th = np.tanh(np.log1p(np.exp(x_clip))) # Tanh de log(1 + exp(x_clip))
        return th + x * sp * (1 - th**2) # Derivada da Mish

    def _softmax(self, x):
        """
        @brief Função softmax para classificação multiclasse.
        @param x: np.ndarray - Entrada.
        @return np.ndarray - Saída após softmax.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Evita overflow subtraindo o máximo
        return e_x / np.sum(e_x, axis=1, keepdims=True) # Normaliza para somar 1 em cada linha

    # --- Normalizações ---
    def _layer_norm(self, x, eps=1e-5):
        """
        @brief Aplica Layer Normalization na saída da camada.
        @param x: np.ndarray - Saída da camada (batch, features)
        @param eps: float - Pequeno valor para estabilidade numérica.
        @return np.ndarray - Saída normalizada.
        """
        mean = np.mean(x, axis=1, keepdims=True) # Média ao longo das features
        std = np.std(x, axis=1, keepdims=True) # Desvio padrão ao longo das features
        return (x - mean) / (std + eps) # Normaliza subtraindo a média e dividindo pelo desvio padrão

    def _init_batch_norm_params(self):
        """
        @brief Inicializa parâmetros de batch norm para cada camada oculta.
        """
        self.bn_gamma = [np.ones((1, h)) for h in self.hidden_layers]  # @var bn_gamma
        self.bn_beta = [np.zeros((1, h)) for h in self.hidden_layers]  # @var bn_beta
        self.bn_running_mean = [np.zeros((1, h)) for h in self.hidden_layers]  # @var bn_running_mean
        self.bn_running_var = [np.ones((1, h)) for h in self.hidden_layers]  # @var bn_running_var
        self.bn_momentum = 0.9  # @var bn_momentum

    def _batch_norm(self, x, idx, training=True, eps=1e-5):
        """
        @brief Aplica Batch Normalization na saída da camada.
        @param x: np.ndarray - Saída da camada (batch, features)
        @param idx: int - Índice da camada oculta.
        @param training: bool - True se em modo treinamento.
        @param eps: float - Pequeno valor para estabilidade numérica.
        @return np.ndarray - Saída normalizada.
        """
        if training:
            mean = np.mean(x, axis=0, keepdims=True) # Calcula média ao longo do batch
            var = np.var(x, axis=0, keepdims=True) # Calcula média e variância
            self.bn_running_mean[idx] = self.bn_momentum * self.bn_running_mean[idx] + (1 - self.bn_momentum) * mean # Atualiza média móvel
            self.bn_running_var[idx] = self.bn_momentum * self.bn_running_var[idx] + (1 - self.bn_momentum) * var # Atualiza variância móvel
        else:
            mean = self.bn_running_mean[idx] # Usa média móvel durante inferência
            var = self.bn_running_var[idx] # Usa variância móvel durante inferência
        x_norm = (x - mean) / np.sqrt(var + eps) # Normaliza subtraindo a média e dividindo pelo desvio padrão
        return self.bn_gamma[idx] * x_norm + self.bn_beta[idx] # Aplica escala e deslocamento

    # --- Inicialização dos pesos ---
    def _init_weights(self):
        """
        @brief Inicializa pesos e biases para cada camada do MLP.
        Utiliza inicialização normal com variância escalada para evitar problemas de gradiente.
        """
        layer_sizes = [self.input_dim] + self.hidden_layers + [self.output_dim]  # @var layer_sizes
        self.weights = []  # @var weights
        self.biases = []   # @var biases
        rng = np.random.default_rng(42) # @var rng
        for i in range(len(layer_sizes) - 1):
            w = rng.normal(0, np.sqrt(2 / layer_sizes[i]), (layer_sizes[i], layer_sizes[i+1]))  # @var w
            b = np.zeros((1, layer_sizes[i+1]))  # @var b
            self.weights.append(w)
            self.biases.append(b)

    # --- Seleção de função de ativação ---
    def _get_activation(self, name):
        """
        @brief Retorna a função de ativação e sua derivada.
        @param name: str - Nome da ativação.
        @return tuple - (função, derivada)
        """
        if name == 'relu':
            return self._relu, self._relu_deriv
        elif name == 'sigmoid':
            return self._sigmoid, self._sigmoid_deriv
        elif name == 'tanh':
            return self._tanh, self._tanh_deriv
        elif name == 'swish':
            return self._swish, self._swish_deriv
        elif name == 'gelu':
            return self._gelu, self._gelu_deriv
        elif name == 'mish':
            return self._mish, self._mish_deriv
        elif name == 'leaky_relu':
            return self._leaky_relu, self._leaky_relu_deriv
        elif name == 'elu':
            return self._elu, self._elu_deriv
        else:
            raise ValueError(f"Ativação desconhecida: {name}")

    # --- Forward ---
    def _forward(self, X, training=True):
        """
        @brief Propagação direta (forward) do MLP.
        @param X: np.ndarray - Dados de entrada (batch_size, n_features).
        @param training: bool - Se True, aplica dropout.
        @return tuple - (ativacoes, zs): ativações e pré-ativações de cada camada.
        """
        a = X  # @var a
        activations = [a]  # @var activations
        zs = []  # @var zs
        for i in range(len(self.hidden_layers)):
            z = np.dot(a, self.weights[i]) + self.biases[i]  # @var z
            zs.append(z)
            act, _ = self._get_activation(self.activation)
            a = act(z)
            # Layer Normalization
            if self.layer_norm:
                a = self._layer_norm(a)
            # Batch Normalization
            if self.batch_norm:
                a = self._batch_norm(a, i, training=training)
            # Dropout apenas durante o treinamento
            if training and self.dropout > 0:
                mask = (np.random.rand(*a.shape) > self.dropout) / (1 - self.dropout)  # @var mask
                a *= mask
            activations.append(a)
        # Camada de saída
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        zs.append(z)
        if self.task == 'classification':
            if self.output_dim == 1:
                a = self._sigmoid(z)
            else:
                a = self._softmax(z)
        else:
            a = z  # Regressão: saída linear
        activations.append(a)
        return activations, zs

    # --- Loss ---
    def _compute_loss(self, y_true, y_pred):
        """
        @brief Calcula a função de perda do MLP.
        @param y_true: np.ndarray - Valores verdadeiros.
        @param y_pred: np.ndarray - Predições da rede.
        @return float - Valor escalar da perda.
        """
        if self.task == 'classification':
            if self.output_dim == 1:
                eps = 1e-8  # @var eps
                return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)) # Loss binária
            else:
                eps = 1e-8  # @var eps
                y_true = y_true.astype(int) # Certifica-se de que y_true é inteiro
                y_pred = np.clip(y_pred, eps, 1 - eps) # Evita overflow
                idx = np.arange(y_true.shape[0])  # @var idx
                return -np.mean(np.log(y_pred[idx, y_true])) # Loss categórica
        else:
            return np.mean((y_true - y_pred) ** 2) # Loss de regressão: MSE (Mean Squared Error)

    # --- Backward ---
    def _backward(self, X, y, activations, zs):
        """
        @brief Retropropagação do erro (backpropagation) do MLP.
        @param X: np.ndarray - Dados de entrada.
        @param y: np.ndarray - Rótulos verdadeiros.
        @param activations: list - Ativações de cada camada.
        @param zs: list - Pré-ativações de cada camada.
        @return tuple - (grads_w, grads_b): gradientes dos pesos e biases.
        """
        grads_w = [np.zeros_like(w) for w in self.weights]  # @var grads_w
        grads_b = [np.zeros_like(b) for b in self.biases]   # @var grads_b
        m = X.shape[0]  # @var m
        if self.task == 'classification':
            if self.output_dim == 1:
                delta = (activations[-1] - y.reshape(-1, 1))  # @var delta
            else:
                y_onehot = np.zeros_like(activations[-1])  # @var y_onehot
                y_onehot[np.arange(m), y.astype(int)] = 1 # Cria vetor one-hot
                delta = (activations[-1] - y_onehot) / m # @var delta
        else:
            delta = (activations[-1] - y.reshape(-1, 1)) # @var delta
        grads_w[-1] = np.dot(activations[-2].T, delta) / m # Gradiente dos pesos da camada de saída
        grads_b[-1] = np.mean(delta, axis=0, keepdims=True) # Gradiente dos biases da camada de saída
        for l in range(len(self.hidden_layers)-1, -1, -1):
            _, deriv = self._get_activation(self.activation) # @var deriv
            delta = np.dot(delta, self.weights[l+1].T) * deriv(zs[l]) # Propaga o erro para a camada anterior
            grads_w[l] = np.dot(activations[l].T, delta) / m # Gradiente dos pesos da camada l
            grads_b[l] = np.mean(delta, axis=0, keepdims=True) # Gradiente dos biases da camada l
        return grads_w, grads_b

    # --- Fit ---
    def fit(self, X, y, X_val=None, y_val=None):
        """
        @brief Treina o MLP usando mini-batch SGD e early stopping.
        @param X: np.ndarray - Dados de treino.
        @param y: np.ndarray - Rótulos de treino.
        @param X_val: np.ndarray - Dados de validação (opcional).
        @param y_val: np.ndarray - Rótulos de validação (opcional).
        """
        X = np.array(X)
        y = np.array(y)
        best_loss = float('inf')  # @var best_loss
        best_weights = None  # @var best_weights
        best_biases = None   # @var best_biases
        epochs_no_improve = 0  # @var epochs_no_improve

        for epoch in range(self.epochs):
            idx = np.random.permutation(len(X))  # @var idx
            X_shuf, y_shuf = X[idx], y[idx]  # @var X_shuf, y_shuf
            for i in range(0, len(X), self.batch_size):
                Xb = X_shuf[i:i+self.batch_size]  # @var Xb
                yb = y_shuf[i:i+self.batch_size]  # @var yb
                activations, zs = self._forward(Xb, training=True) # @var activations, zs
                grads_w, grads_b = self._backward(Xb, yb, activations, zs) # @var grads_w, grads_b
                for l in range(len(self.weights)):
                    self.weights[l] -= self.lr * grads_w[l] # Atualiza pesos
                    self.biases[l] -= self.lr * grads_b[l] # Atualiza biases
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)  # @var val_pred
                val_loss = self._compute_loss(y_val, val_pred)  # @var val_loss
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - val_loss: {val_loss:.4f}") # Exibe perda de validação
                if val_loss < best_loss - 1e-6:
                    best_loss = val_loss # Atualiza melhor perda
                    best_weights = [w.copy() for w in self.weights] # Copia pesos
                    best_biases = [b.copy() for b in self.biases] # Copia biases
                    epochs_no_improve = 0 # Reseta contador de épocas sem melhora
                else:
                    epochs_no_improve += 1 # Incrementa contador de épocas sem melhora
                    if epochs_no_improve >= self.patience:
                        if self.verbose:
                            print("Early stopping!") # Exibe mensagem de early stopping
                        break
            else:
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} concluído.")
        if best_weights is not None:
            self.weights = best_weights # Restaura melhores pesos
            self.biases = best_biases # Restaura melhores biases
        self.is_trained = True # Marca como treinado

    # --- Predict ---
    def predict(self, X):
        """
        @brief Realiza predições com o MLP treinado.
        @param X: np.ndarray - Dados de entrada.
        @return np.ndarray - Predições.
        """
        X = np.array(X) # Garante que X é um array numpy
        activations, _ = self._forward(X, training=False) # @var activations
        out = activations[-1] # Saída da última camada
        if self.task == 'classification':
            if self.output_dim == 1:
                return (out.flatten() > 0.5).astype(int) # Retorna 0 ou 1 para classificação binária
            else:
                return np.argmax(out, axis=1) # Retorna o índice da classe mais provável para classificação multiclasse
        else:
            return out.flatten() # Retorna o valor contínuo para regressão

    # --- Score ---
    def score(self, X, y, metric):
        """
        @brief Avalia o modelo usando uma métrica fornecida.
        @param X: np.ndarray - Dados de entrada.
        @param y: np.ndarray - Rótulos verdadeiros.
        @param metric: function - Função de métrica.
        @return float - Resultado da métrica.
        """
        y_pred = self.predict(X) # @var y_pred
        return metric(y, y_pred) # Retorna o resultado da métrica


    # --- Checagem de treinamento ---
    def is_fitted(self):
        """
        @brief Verifica se o modelo já foi treinado.
        @return bool - True se treinado, False caso contrário.
        """
        return self.is_trained 
