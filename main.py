
from get_data import *
from Indices_fin import *
from Senales_cv import *

from standarizacion_datos import *
from modelo_cnn import *
from Modelos import *
from evaluacion_modelos import *
def main():
    print("Iniciando flujo de procesamiento...")

    # Parámetros generales del proceso
    w = 30
    ticker = 'ORCL'

    print("Obteniendo datos del activo:", ticker)
    df = get_data(ticker)

    print("Calculando índices financieros...")
    df = calcular_indices(df, w)

    print("Generando señales del trading system...")
    df_real = generar_senales(df)

    print("⚖️ Estandarizando datos...")
    df_estandar = standarizacion_datos(df_real)

    print(" Entrenando modelo MLP...")
    m_mlp = modelo_mlp(df_estandar, 'signal', 50, 32, 0.1, 0.6, 0.2,
                       'drift_report_mlp', w, 6, 3)

    print("Entrenando modelo CNN...")
    m_cnn = modelo_cnn(df_estandar, 'signal', 50, 32, 0.1, 0.6, 0.2,
                       'drift_report_cnn', w, 6, 3)

    print(" Cargando rutas a modelos (MLflow artifacts)...")
    ruta_mlp = ('file:///C:/Users/52331/PycharmProjects/PythonProject4/'
                'Proyecto-3/mlruns/178340679539533950/models/'
                'm-0b8695dbd759460981fa0c73f11eeb48/artifacts')

    ruta_cnn = ('file:///C:/Users/52331/PycharmProjects/PythonProject4/'
                'Proyecto-3/mlruns/560905997404354661/models/'
                'm-3183024883b64e599cdcdb94d5546fb7/artifacts')

    print(" Evaluando modelo MLP...")
    r_df_real, met_real, r_df_mlp, met_mlp = eval_model(
        df_estandar, df_real, ruta_mlp, 0.8, 'signal',
        100000, 0.0125, 10, 0.05, 0.05, 0.0025, 0.2, 1
    )

    print(" Evaluando modelo CNN...")
    _, _, r_df_cnn, met_cnn = eval_model(
        df_estandar, df_real, ruta_cnn, 0.8, 'signal',
        100000, 0.0125, 10, 0.05, 0.05, 0.0025, 0.2, 2
    )

    print("Proceso completado con éxito.")
    print(" Resultados:")
    print("   - Métricas MLP:", met_mlp)
    print("   - Métricas CNN:", met_cnn)


if __name__ == "__main__":
    main()