import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class Comparator:
    @staticmethod
    def rmse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.shape != y_pred.shape or y_true.size == 0:
            return float('nan')
        mse = np.mean((y_true - y_pred) ** 2)
        return np.sqrt(mse)

    @staticmethod
    def precision(y_true, y_pred, average='binary'):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        if average == 'binary':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true != 1) & (y_pred == 1))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif average in ['macro', 'weighted']:
            precisions = []
            weights = []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                precisions.append(p)
                weights.append(np.sum(y_true == cls))
            if average == 'macro':
                return np.mean(precisions)
            else:
                return np.average(precisions, weights=weights)
        elif average == 'micro':
            tp = sum(np.sum((y_true == cls) & (y_pred == cls)) for cls in classes)
            fp = sum(np.sum((y_true != cls) & (y_pred == cls)) for cls in classes)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            raise ValueError("average must be 'binary', 'macro', 'micro', or 'weighted'")

    @staticmethod
    def recall(y_true, y_pred, average='binary'):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        if average == 'binary':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred != 1))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif average in ['macro', 'weighted']:
            recalls = []
            weights = []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                recalls.append(r)
                weights.append(np.sum(y_true == cls))
            if average == 'macro':
                return np.mean(recalls)
            else:
                return np.average(recalls, weights=weights)
        elif average == 'micro':
            tp = sum(np.sum((y_true == cls) & (y_pred == cls)) for cls in classes)
            fn = sum(np.sum((y_true == cls) & (y_pred != cls)) for cls in classes)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            raise ValueError("average must be 'binary', 'macro', 'micro', or 'weighted'")

    @staticmethod
    def f_measure(y_true, y_pred, average='binary'):
        prec = Comparator.precision(y_true, y_pred, average)
        rec = Comparator.recall(y_true, y_pred, average)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    @staticmethod
    def accuracy(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(y_true == y_pred)

    @staticmethod
    def confusion(y_true, y_pred, labels=None):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for i, cls_true in enumerate(labels):
            for j, cls_pred in enumerate(labels):
                matrix[i, j] = np.sum((y_true == cls_true) & (y_pred == cls_pred))
        return matrix

    @staticmethod
    def classification_report(y_true, y_pred, digits=3):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        report = "Class\tPrec\tRec\tF1\tSupport\n"
        for cls in classes:
            prec = Comparator.precision(y_true == cls, y_pred == cls)
            rec = Comparator.recall(y_true == cls, y_pred == cls)
            f1 = Comparator.f_measure(y_true == cls, y_pred == cls)
            support = np.sum(y_true == cls)
            report += f"{cls}\t{prec:.{digits}f}\t{rec:.{digits}f}\t{f1:.{digits}f}\t{support}\n"
        acc = Comparator.accuracy(y_true, y_pred)
        report += f"\nAccuracy: {acc:.{digits}f}\n"
        return report

    @staticmethod
    def compare_models_with_reference(preds_dir, reference_path, result_dir, label_col="classe"):
        """
        Compara cada solução (arquivo de predição) com o gabarito e gera gráficos de precisão, recall, f1, rmse, acurácia e matriz de confusão.
        """
        df_ref = pd.read_csv(reference_path)
        if label_col not in df_ref.columns:
            raise ValueError(f"Coluna '{label_col}' não encontrada no arquivo de referência.")
        y_true = df_ref[label_col].values

        pred_files = [f for f in os.listdir(preds_dir) if f.endswith("_pred.csv")]
        acc_dict = {}
        prec_dict = {}
        recall_dict = {}
        f1_dict = {}
        rmse_dict = {}

        for file in pred_files:
            model_name = file.replace("_pred.csv", "")
            df_pred = pd.read_csv(os.path.join(preds_dir, file))
            if "prediction" not in df_pred.columns:
                continue
            y_pred = df_pred["prediction"].values
            min_len = min(len(y_true), len(y_pred))
            yt = y_true[:min_len]
            yp = y_pred[:min_len]
            acc_dict[model_name] = Comparator.accuracy(yt, yp)
            prec_dict[model_name] = Comparator.precision(yt, yp, average='macro')
            recall_dict[model_name] = Comparator.recall(yt, yp, average='macro')
            f1_dict[model_name] = Comparator.f_measure(yt, yp, average='macro')
            rmse_dict[model_name] = Comparator.rmse(yt, yp)

            # Matriz de confusão individual
            conf = Comparator.confusion(yt, yp)
            plt.figure(figsize=(5, 4))
            plt.imshow(conf, cmap='Blues', interpolation='nearest')
            plt.title(f"Matriz de Confusão - {model_name}")
            plt.xlabel("Previsto")
            plt.ylabel("Verdadeiro")
            plt.colorbar()
            tick_marks = np.arange(len(np.unique(y_true)))
            plt.xticks(tick_marks, np.unique(y_true), rotation=45)
            plt.yticks(tick_marks, np.unique(y_true))
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f"confusion_{model_name}.png"))
            plt.close()

        # Gera gráficos comparativos de métricas
        def plot_metric(metric_dict, metric_name, ylabel):
            n = len(metric_dict)
            plt.figure(figsize=(max(14, n * 1.2), 12))  # height bem maior
            plt.bar(metric_dict.keys(), metric_dict.values())
            plt.ylabel(ylabel)
            plt.xlabel("Modelo")
            plt.title(f"{metric_name} dos modelos comparados ao gabarito")
            plt.xticks(rotation=90)
            plt.subplots_adjust(bottom=0.6)  # margem inferior bem maior
            plt.tight_layout()  # sem rect
            plt.savefig(os.path.join(result_dir, f"comparacao_{metric_name.lower()}.png"))
            plt.close()

        plot_metric(acc_dict, "acuracia", "Acurácia")
        plot_metric(prec_dict, "precisao", "Precisão")
        plot_metric(recall_dict, "recall", "Recall")
        plot_metric(f1_dict, "f1-score", "F1-score")
        plot_metric(rmse_dict, "rmse", "RMSE")

        return {
            "accuracy": acc_dict,
            "precision": prec_dict,
            "recall": recall_dict,
            "f1": f1_dict,
            "rmse": rmse_dict
        }
