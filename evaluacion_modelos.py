import mlflow
from backtesting import *


def eval_model(df_estandar,df_real,ruta_m,tv,t_col,cash,com,n_shares,sl,tp,br,ct,l_b):

    #generamos la ruta del modelo
    model_uri = ruta_m
    modelo_pred = mlflow.pyfunc.load_model(model_uri)



    # Asignamos la variable a estandarizar
    target_col = t_col


    # División cronológica: 60% train, 20% test, 20% val
    df_val = df_estandar.iloc[int(len(df_estandar) * tv):]
    df_val_pred = df_val.drop(columns = target_col)
    X_val = df_val_pred.values
    # verificamos que sea modelo mlp o cnn para ver de que forma entregamos los valores de validacion para intentar entrenar el modelo
    if l_b == 1:
        X_val = X_val
    elif l_b == 2:
        X_val = np.expand_dims(X_val, axis=-1)
    else:
        raise ValueError("El parámetro l_b debe ser 1 (mlp) o 2 (cnn).")
    p = modelo_pred.predict(X_val)
    df_pred = df_real.iloc[int(len(df_real) * 0.8):]
    df_pred = df_pred.drop(columns = target_col)
    # igual aqui vemos que tipo de modelo es para ver de que manera redondeamos los datos ya sea redondeo o el que tiene el valor mas alto para el cnn
    if l_b ==1:
        con = [
            p<0,
            p<0.5,
            p>0.51
        ]
        ele = [
            -1,
            0,
            1
        ]
        p_red = np.select(con, ele)
    elif l_b == 2:
        p_red = np.argmax(p, axis=1)

    df_pred['signal'] = p_red


    # ya se usa la funcion de backtesting para verificar el exito del modelo comparado contra el valor real
    r_df_real, met_real= backtest(df_real,cash = cash,com = com,n_shares =n_shares,sl=sl,tp=tp,borrow_rate=br,cash_threshold=ct)

    r_df_mod, met_mod = backtest(df_pred,cash = cash,com = com,n_shares =n_shares,sl=sl,tp=tp,borrow_rate=br,cash_threshold=ct)

    return r_df_real, met_real, r_df_mod, met_mod

