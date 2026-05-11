def pysb(
    y,
    min_pct=0.15,
    alpha=0.05,
    max_breaks=None,
    plot=True
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import chi2
    import statsmodels.api as sm

    """
    Detecta múltiplas quebras estruturais em uma série temporal usando
    Supremum Wald Test com Binary Segmentation.

    Parâmetros
    ----------
    y : pd.Series
        Série temporal com índice datetime.

    min_pct : float
        Percentual mínimo de observações antes e depois de cada possível quebra.

    alpha : float
        Nível de significância para aceitar uma quebra estrutural.

    max_breaks : int ou None
        Número máximo de quebras estruturais permitidas.
        Se None, detecta enquanto houver quebras significativas.

    plot : bool
        Se True, plota a série com as quebras detectadas.

    Retorna
    -------
    results : pd.DataFrame
        DataFrame com as quebras detectadas, estatística Wald e p-valor.
    """

    # ============================================================
    # Validações iniciais
    # ============================================================

    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series with a datetime index")

    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("y must have a datetime index")

    y = y.dropna().copy()
    y = y.sort_index()

    dates = y.index
    y_values = y.values

    # ============================================================
    # Função interna: detecta 1 break em uma subamostra
    # ============================================================

    def single_sup_wald(y_sub, offset=0):
        """
        Aplica Supremum Wald Test em uma subamostra.
        Retorna a melhor quebra dentro daquela subamostra.
        """

        n = len(y_sub)

        min_obs = int(np.floor(n * min_pct))
        max_obs = n - min_obs

        if min_obs < 2 or max_obs <= min_obs:
            return None

        wald_stats = []
        breakpoints = []

        model_full = sm.OLS(y_sub, np.ones(n)).fit()
        RSS_full = np.sum(model_full.resid ** 2)

        k = 1

        for b in range(min_obs, max_obs):

            y1 = y_sub[:b]
            y2 = y_sub[b:]

            if len(y1) <= k or len(y2) <= k:
                continue

            model1 = sm.OLS(y1, np.ones(len(y1))).fit()
            model2 = sm.OLS(y2, np.ones(len(y2))).fit()

            RSS1 = np.sum(model1.resid ** 2)
            RSS2 = np.sum(model2.resid ** 2)

            RSS_total = RSS1 + RSS2

            denominator = RSS_total / (n - 2 * k)

            if denominator == 0:
                continue

            wald_stat = ((RSS_full - RSS_total) / k) / denominator

            wald_stats.append(wald_stat)
            breakpoints.append(b)

        if not wald_stats:
            return None

        sup_wald_stat = max(wald_stats)
        local_break_index = breakpoints[wald_stats.index(sup_wald_stat)]
        global_break_index = offset + local_break_index

        p_value = 1 - chi2.cdf(sup_wald_stat, df=k)

        return {
            "break_index": global_break_index,
            "local_break_index": local_break_index,
            "break_date": dates[global_break_index],
            "sup_wald_stat": sup_wald_stat,
            "p_value": p_value,
            "segment_start_index": offset,
            "segment_end_index": offset + n - 1,
            "segment_start_date": dates[offset],
            "segment_end_date": dates[offset + n - 1],
            "n_segment": n
        }

    # ============================================================
    # Função interna: binary segmentation recursivo
    # ============================================================

    detected_breaks = []

    def recursive_search(start_idx, end_idx):
        """
        Busca quebras em um intervalo da série.
        """

        if max_breaks is not None and len(detected_breaks) >= max_breaks:
            return

        y_sub = y_values[start_idx:end_idx + 1]

        n_sub = len(y_sub)

        min_required = max(10, int(2 / min_pct))

        if n_sub < min_required:
            return

        result = single_sup_wald(y_sub, offset=start_idx)

        if result is None:
            return

        if result["p_value"] < alpha:

            detected_breaks.append(result)

            b = result["break_index"]

            recursive_search(start_idx, b - 1)
            recursive_search(b, end_idx)

    # ============================================================
    # Executa busca recursiva
    # ============================================================

    recursive_search(0, len(y_values) - 1)

    # ============================================================
    # Organiza resultados
    # ============================================================

    if detected_breaks:
        results = pd.DataFrame(detected_breaks)

        results = results.sort_values("break_date").reset_index(drop=True)

        results = results[
            [
                "break_date",
                "break_index",
                "sup_wald_stat",
                "p_value",
                "segment_start_date",
                "segment_end_date",
                "n_segment"
            ]
        ]

    else:
        results = pd.DataFrame(
            columns=[
                "break_date",
                "break_index",
                "sup_wald_stat",
                "p_value",
                "segment_start_date",
                "segment_end_date",
                "n_segment"
            ]
        )

    # ============================================================
    # Plot
    # ============================================================

    if plot:
        plt.figure(figsize=(12, 6))

        plt.plot(
            dates,
            y_values,
            label="Time Series",
            #marker="--"
        )

        if not results.empty:
            for _, row in results.iterrows():
                plt.axvline(
                    row["break_date"],
                    linestyle="--",
                    label=f"Break: {row['break_date'].date()}"
                )

        plt.title("Time Series with Multiple Structural Breaks")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    return results