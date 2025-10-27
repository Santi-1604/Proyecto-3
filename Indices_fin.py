import numpy as np
import pandas as pd
import ta
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calcular_indices(df, w=10):
    df = df.copy()

    # 1. ROC
    df['Ind_roc'] = (df['Close'] - df['Close'].shift(w - 1)) / w

    # 2. RSI
    rsi_indicador = ta.momentum.RSIIndicator(df['Close'], window=w)
    df['Ind_rsi'] = rsi_indicador.rsi()

    # 3. Williams %R
    williamr = ta.momentum.williams_r(
        close=df['Close'],
        low=df['Low'],
        high=df['High'],
        lbp=w
    )
    df['Ind_willr'] = williamr

    # 4. KAMA
    df['Ind_kama'] = ta.momentum.kama(df['Close'], window=w)

    # 5. Awesome Oscillator
    A_o = ta.momentum.AwesomeOscillatorIndicator(
        df['High'], df['Low'], window1=int(np.round(w/2)), window2=w
    )
    df['Ind_A0'] = A_o.awesome_oscillator()

    # 6. CMF
    Mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    Mfv = Mfm * df['Volume']
    df['Ind_cmf'] = Mfv.rolling(w).sum() / df['Volume'].rolling(w).sum()

    # 7. Choppiness Index
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=w).sum()
    m = high.rolling(window=w).max() - low.rolling(window=w).min()
    df['Ind_chop'] = 100 * (np.log10(atr_sum / m) / np.log10(w))

    # 8. Volatility Ratio
    atr_m_short = tr.rolling(window=w).mean()
    atr_m_long = tr.rolling(window=w + 10).mean()
    df['Ind_vr'] = atr_m_short / atr_m_long

    # 9. Bollinger Bands
    boll = ta.volatility.BollingerBands(close=df['Close'], window=w, window_dev=int(np.round(w/5)))
    df['Ind_boll_h'] = boll.bollinger_hband()
    df['Ind_boll_m'] = boll.bollinger_mavg()
    df['Ind_boll_l'] = boll.bollinger_lband()

    # 10. Coeficiente de variación
    rend = np.log(df['Close'] / df['Close'].shift(1))
    sigma_window = rend.rolling(window=w).std()
    df['Ind_cdv'] = rend / sigma_window

    # 11. OBV
    ob = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], volume=df['Volume']
    )
    df['Ind_obv'] = ob.on_balance_volume()

    # 12. Ease of Movement
    eom = ta.volume.EaseOfMovementIndicator(
        high=df['High'],
        low=df['Low'],
        volume=df['Volume'],
        window=w
    )
    df['Ind_eom'] = eom.ease_of_movement()

    # 13. Force Index
    fi = ta.volume.ForceIndexIndicator(
        close=df['Close'], volume=df['Volume'], window=w
    )
    df['Ind_fi'] = fi.force_index()

    # 14. VWAP
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=w
    )
    df['Ind_vwap'] = vwap.volume_weighted_average_price()

    # 15. Keltner Channel
    kc = ta.volatility.KeltnerChannel(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=w,
        window_atr=int(np.round(w / 2)),
        original_version=False
    )
    df['Ind_kc_h'] = kc.keltner_channel_hband()
    df['Ind_kc_m'] = kc.keltner_channel_mband()
    df['Ind_kc_l'] = kc.keltner_channel_lband()

    # 16. CMO
    high_s = high.rolling(window=w).sum()
    low_s = low.rolling(window=w).sum()
    df['Ind_cmo'] = (high_s - low_s) / (high_s + low_s) * 100

    # 17. MFI
    df['Ind_mfi'] = ta.volume.money_flow_index(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=w
    )

    # 18. ADI
    adi = ta.volume.AccDistIndexIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume']
    )
    df['Ind_adi'] = adi.acc_dist_index()

    # 19. Ulcer Index
    df['Ind_ui'] = ta.volatility.ulcer_index(df['Close'], window=14)
    df = df.dropna()

    # 20. Regímenes
    prices = df['Close']
    if isinstance(prices, pd.Series):
        series = prices.copy()
    else:
        series = prices.iloc[:, 0].copy()

    # Retornos logarítmicos
    rets = np.log(series / series.shift(1)).dropna()

    # Cálculo de características en ventana móvil
    features, idxs = [], []
    for end in range(w, len(rets) + 1):
        r = rets.iloc[end - w:end]
        momentum = r.mean()
        vol = r.std()
        autocorr = r.autocorr(lag=1)
        sign_changes = np.sum(np.sign(r.values[1:]) != np.sign(r.values[:-1])) / (len(r) - 1)
        features.append([momentum, autocorr, vol, sign_changes])
        idxs.append(r.index[-1])

    features_df = pd.DataFrame(features, index=idxs, columns=['momentum', 'autocorr', 'vol', 'reversals'])

    # Escalado y clustering
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[['momentum', 'autocorr', 'vol', 'reversals']])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(X)

    features_df['cluster'] = clusters

    # Prototipos heurísticos
    prototypes = {
        'HighMomentum': np.array([+1.5, +0.5, 0.0, -1.0]),
        'MeanReverting': np.array([-0.5, -1.0, -0.3, +1.2]),
        'Crisis': np.array([-1.0, +0.7, +1.5, +0.5])
    }

    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return (a @ b) / denom if denom != 0 else -1

    # Asignación de etiquetas numéricas
    label_map = {'Crisis': 0, 'HighMomentum': 1, 'MeanReverting': 2}

    centroid_labels = []
    for c in kmeans.cluster_centers_:
        sims = {k: cosine(c, prototypes[k]) for k in prototypes}
        best_label = max(sims.items(), key=lambda x: x[1])[0]
        centroid_labels.append(best_label)

    cluster_to_regime = {i: label_map[centroid_labels[i]] for i in range(len(centroid_labels))}
    features_df['regime'] = features_df['cluster'].map(cluster_to_regime)

    # Agregar al DataFrame original
    df['Ind_reg'] = features_df['regime']
    sma = prices.rolling(window=w).mean()
    df['sma'] = sma
    df = df.dropna()
    return df

