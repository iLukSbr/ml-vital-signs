import pandas as pd

class DatasetManager:
    """
    Classe utilitária para operações de carregamento e alinhamento de datasets.
    """

    @staticmethod
    def load_train_dataset(path, target_column):
        """
        Carrega um dataset de treino a partir de um arquivo CSV.

        @param path Caminho para o arquivo CSV.
        @param target_column Nome da coluna alvo (classe).
        @return DataFrame de features (com nomes das colunas), array de classes.
        """
        df = pd.read_csv(path)
        X = df.drop(columns=[target_column])  # DataFrame de features
        y = df[target_column].values          # Array de classes
        return X, y

    @staticmethod
    def load_predict_dataset(path):
        """
        Carrega um dataset de predição a partir de um arquivo CSV.

        @param path Caminho para o arquivo CSV.
        @return DataFrame com os dados de predição (pode conter ou não a coluna de classe).
        """
        df = pd.read_csv(path)
        return df

    @staticmethod
    def combine_train_datasets(dfs):
        """
        Alinha as colunas de múltiplos DataFrames de treino, preenchendo valores ausentes com pd.NA.

        @param dfs Lista de DataFrames de treino.
        @return DataFrame único com todas as colunas presentes em pelo menos um DataFrame, valores ausentes como pd.NA.
        """
        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)  # Junta todas as colunas presentes em qualquer DataFrame
        all_columns = sorted(list(all_columns))
        dfs_aligned = [df.reindex(columns=all_columns, fill_value=pd.NA) for df in dfs]  # Alinha colunas e preenche ausentes
        combined = pd.concat(dfs_aligned, ignore_index=True)  # Concatena todos os DataFrames alinhados
        return combined

    @staticmethod
    def align_features(X_train_df, X_pred_df):
        """
        Alinha as colunas do dataset de predição com as do treino.
        Colunas ausentes são preenchidas com pd.NA, colunas extras são descartadas.

        @param X_train_df DataFrame de treino (com nomes das colunas).
        @param X_pred_df DataFrame de predição (com nomes das colunas).
        @return np.ndarray com as features alinhadas para predição.
        """
        X_pred_aligned = X_pred_df.reindex(columns=X_train_df.columns, fill_value=pd.NA)
        return X_pred_aligned.values
