from sklearn.preprocessing import StandardScaler
import numpy as np



def standarizacion_datos(df):

    price_cols = ['Close', 'High', 'Low', 'Open']
    volume_cols = ['Volume']
    indicator_cols = [c for c in df.columns if c.startswith('Ind_')]
    binary_cols = ['coseno_compra', 'coseno_venta']

    # Copia
    df_scaled = df.copy()

    # Escalar precios (mantener relación entre ellos)
    scaler_prices = StandardScaler()
    df_scaled[price_cols] = scaler_prices.fit_transform(df[price_cols])

    # Escalar volumen (log + z-score)
    scaler_vol = StandardScaler()
    df_scaled[volume_cols] = scaler_vol.fit_transform(np.log1p(df[volume_cols]))

    # Escalar indicadores técnicos
    scaler_ind = StandardScaler()
    df_scaled[indicator_cols] = scaler_ind.fit_transform(df[indicator_cols])

    # Mantener binarios igual
    df_scaled[binary_cols] = df[binary_cols].astype(float)
    df_scaled = df_scaled.dropna()
    return(df_scaled)