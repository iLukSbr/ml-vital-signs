import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from machine_learning import SymbolicRandomForest, SymbolicC45, NeuralNetwork, DatasetManager, Comparator

##
# @brief Função principal do programa. Treina, prediz e salva resultados dos modelos.
#
# @details
# - Carrega múltiplos datasets de treino, combina-os alinhando as colunas.
# - Treina vários modelos de classificação sobre o conjunto combinado.
# - Realiza predição sobre um dataset de predição (sem coluna de classe).
# - Salva as predições de cada modelo em arquivos CSV na pasta de resultados.
#
# @note
# Os arquivos de treino devem conter a coluna de classe (target).
# O arquivo de predição não deve conter a coluna de classe.
def main():
    # Lista de dicionários com os datasets de treino.
    # @var train_datasets
    # @type list[dict]
    # @key path   Caminho para o arquivo CSV de treino.
    # @key target Nome da coluna de classe (target) no arquivo de treino.
    train_datasets = [
        {"path": "dataset/train/treino_sinais_vitais_com_label.csv", "target": "classe"},
    ]

    # Dicionário com o dataset de predição (sem coluna de classe).
    # @var predict_dataset
    # @type dict
    # @key path Caminho para o arquivo CSV de predição.
    predict_dataset = {
        "path": "dataset/test/sinaisvitais_teste.csv"
    }
    
    reference_path = "dataset/test/sinaisvitais_gabarito.csv"
    label_col = "classe"

    # Cria a pasta de resultados se não existir.
    # @var result_dir
    # @type str
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    # Carrega e combina os datasets de treino.
    # @var X_trains Lista de DataFrames de features de cada dataset de treino.
    # @var y_trains Lista de arrays de classes de cada dataset de treino.
    X_trains = []
    y_trains = []
    for ds in train_datasets:
        # @param ds["path"] Caminho do arquivo de treino.
        # @param ds["target"] Nome da coluna de classe.
        X, y = DatasetManager.load_train_dataset(ds["path"], ds["target"])
        X_trains.append(X)
        y_trains.append(y)
    # DataFrame único com todas as features dos datasets de treino (colunas alinhadas).
    # @var X_train_df
    X_train_df = DatasetManager.combine_train_datasets(X_trains)
    # Array único com todas as classes dos datasets de treino.
    # @var y_train
    y_train = np.concatenate(y_trains)
    y_offset = np.min(y_train)
    y_train = y_train - y_offset

    # Carrega o dataset de predição.
    # @var X_pred_df DataFrame de predição (pode ter qualquer subconjunto de colunas).
    X_pred_df = DatasetManager.load_predict_dataset(predict_dataset["path"])
    # np.ndarray de features alinhadas para predição.
    # @var X_pred
    X_pred = DatasetManager.align_features(X_train_df, X_pred_df)
    # np.ndarray de treino (para uso nos modelos).
    # @var X_train
    X_train = X_train_df.values

    # Número de classes distintas no treino.
    # @var n_classes
    n_classes = len(np.unique(y_train))
    # Dicionário {nome: modelo} com todos os modelos a serem treinados e avaliados.
    # @var models
    models = {
        # Random Forest
        "Random Forest Simbólico (10 árvores, max_depth=5)": SymbolicRandomForest(
            n_estimators=10,
            max_depth=5,
            random_state=42
        ),
        "Random Forest Simbólico (50 árvores, max_depth=5)": SymbolicRandomForest(
            n_estimators=50,
            max_depth=5,
            random_state=42
        ),
        "Random Forest Simbólico (10 árvores, max_depth=8)": SymbolicRandomForest(
            n_estimators=10,
            max_depth=8,
            random_state=42
        ),
        "Random Forest Simbólico (10 árvores, max_depth=5, sem bootstrap)": SymbolicRandomForest(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            bootstrap=False
        ),

        # C4.5
        "C4.5 (max_depth=10, min_samples_split=5, prune=True)": SymbolicC45(
            max_depth=10,
            min_samples_split=5,
            prune=True
        ),
        "C4.5 (max_depth=5, min_samples_split=5, prune=True)": SymbolicC45(
            max_depth=5,
            min_samples_split=5,
            prune=True
        ),
        "C4.5 (max_depth=10, min_samples_split=2, prune=True)": SymbolicC45(
            max_depth=10,
            min_samples_split=2,
            prune=True
        ),
        "C4.5 (max_depth=10, min_samples_split=5, prune=False)": SymbolicC45(
            max_depth=10,
            min_samples_split=5,
            prune=False
        ),

        # Rede Neural
        "Rede Neural (ReLU, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='relu',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (Tanh, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='tanh',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (Sigmoid, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='sigmoid',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (Swish, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='swish',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (GELU, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='gelu',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (Mish, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='mish',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (Leaky ReLU, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='leaky_relu',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (ELU, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='elu',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (ReLU, dropout=0.5, batch_norm=True, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='relu',
            dropout=0.5,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (ReLU, dropout=0.2, batch_norm=False, layer_norm=False, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='relu',
            dropout=0.2,
            batch_norm=False,
            layer_norm=False,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (ReLU, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.0001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='relu',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-4,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (ReLU, dropout=0.2, batch_norm=True, layer_norm=False, lr=0.001, batch_size=128)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='relu',
            dropout=0.2,
            batch_norm=True,
            layer_norm=False,
            lr=1e-3,
            batch_size=128,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        ),
        "Rede Neural (ReLU, dropout=0.2, batch_norm=True, layer_norm=True, lr=0.001, batch_size=64)": NeuralNetwork(
            input_dim=X_train.shape[1],
            output_dim=1 if n_classes == 2 else n_classes,
            hidden_layers=[128, 64, 32],
            activation='relu',
            dropout=0.2,
            batch_norm=True,
            layer_norm=True,
            lr=1e-3,
            batch_size=64,
            epochs=100,
            patience=10,
            task='classification',
            verbose=False
        )
    }

    # Treina e prediz para cada modelo.
    result_files = {}
    for name, model in models.items():
        print(f"\nTreinando modelo: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        if 'Rede Neural' in name or 'C4.5' in name or 'Random Forest' in name:
            y_pred = y_pred + y_offset
        print(f"Predições para {name} no arquivo {predict_dataset['path']}:")
        print(y_pred)
        pred_filename = os.path.join(
            result_dir,
            f"{os.path.splitext(os.path.basename(predict_dataset['path']))[0]}_{name}_pred.csv"
        )
        pd.DataFrame({"prediction": y_pred}).to_csv(pred_filename, index=False)
        result_files[name] = pred_filename

    # Comparação
    Comparator.compare_models_with_reference(
        preds_dir=result_dir,
        reference_path=reference_path,
        result_dir=result_dir,
        label_col=label_col
    )

    print("Gráficos e comparações salvos em:", result_dir)

if __name__ == "__main__":
    main()
