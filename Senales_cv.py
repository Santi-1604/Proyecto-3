import numpy as np
import pandas as pd


def generar_senales(df):
    senales = pd.DataFrame(index=df.index)

    # Momentum / Volumen / Volatilidad
    senales['roc'] = np.where(df['Ind_roc'] > 0, 1, -1)
    senales['rsi'] = np.where(df['Ind_rsi'] < 20, 1, np.where(df['Ind_rsi'] > 70, -1, 0))
    senales['willr'] = np.where(df['Ind_willr'] < -85, 1, np.where(df['Ind_willr'] > -20, -1, 0))
    senales['A0'] = np.where(df['Ind_A0'] > 0.05, 1, -1)
    senales['cmf'] = np.where(df['Ind_cmf'] > 0.05, 1, -1)
    senales['vr'] = np.where(df['Ind_vr'] > 1, 1, -1)
    senales['mfi'] = np.where(df['Ind_mfi'] < 20, 1, np.where(df['Ind_mfi'] > 80, -1, 0))
    senales['obv'] = np.where(df['Ind_obv'] > df['Ind_obv'].shift(1), 1, -1)
    senales['eom'] = np.where(df['Ind_eom'] > 0, 1, -1)
    senales['fi'] = np.where(df['Ind_fi'] > 0, 1, -1)
    senales['cmo'] = np.where(df['Ind_cmo'] > 0, 1, -1)
    senales['adi'] = np.where(df['Ind_adi'] > df['Ind_adi'].shift(1), 1, -1)
    senales['ui'] = np.where(df['Ind_ui'] < df['Ind_ui'].rolling(5).mean(), 1, -1)
    senales['chop'] = np.where(df['Ind_chop'] < 38, 1, np.where(df['Ind_chop'] > 61.8, -1, 0))
    senales['cdv'] = np.where(df['Ind_cdv'] > 0, 1, -1)
    kama = []
    vwap = []
    boll = []
    kc = []
    reg= []
    for i in range (len(df)):
        kama.append(np.where(df['Close'].iloc[i] > df['Ind_kama'].iloc[i], 1, -1))
        vwap.append(np.where(df['Close'].iloc[i] < df['Ind_vwap'].iloc[i], 1, -1))
        boll.append(np.where(
            df['Close'].iloc[i] < df['Ind_boll_l'].iloc[i], 1,
            np.where(df['Close'].iloc[i] > df['Ind_boll_h'].iloc[i], -1, 0)
        ))

        kc.append(np.where(
            df['Close'].iloc[i] < df['Ind_kc_l'].iloc[i], 1,
            np.where(df['Close'].iloc[i] > df['Ind_kc_h'].iloc[i],-1,0)
        ))

        if df['Ind_reg'].iloc[i] == 1:
            reg.append(np.where(df['Close'].iloc[i] < df['sma'].iloc[i], 1, -1))
        elif df['Ind_reg'].iloc[i] == 2:
            reg.append(np.where(df['Close'].iloc[i] > df['sma'].iloc[i], 1, -1))
        else:
            reg.append(0)

    senales['kama'] = kama
    senales['vwap'] = vwap
    senales['boll'] = boll
    senales['kc'] = kc
    senales['reg'] = reg

    # Calcular consenso
    senales['consenso_compra'] = (senales == 1).sum(axis=1) / len(senales.columns)
    senales['consenso_venta'] = (senales == -1).sum(axis=1) / len(senales.columns)

    senales['Signal'] = np.where(
        senales['consenso_compra'] >= 0.65, 1,
        np.where(senales['consenso_venta'] >= 0.65, -1, 0)
    )
    df['coseno_compra']= senales['consenso_compra']
    df['coseno_venta']= senales['consenso_venta']
    df['signal']= senales['Signal']
    return df
