import pandas as pd
import numpy as np
from dataclasses import dataclass

def backtest(df, cash, com, n_shares, sl, tp, borrow_rate=0.0, cash_threshold=0.01):
    """
    Backtest con borrow rate, stop loss, take profit y control de liquidez.

    Parámetros:
    - df: DataFrame con columnas ['Date', 'Close', 'signal']
    - cash: capital inicial
    - com: comisión (por operación)
    - n_shares: número fijo de acciones a operar (0 = usar todo el capital disponible)
    - sl: stop loss (proporcional, ej. 0.05)
    - tp: take profit (proporcional, ej. 0.05)
    - borrow_rate: tasa de financiamiento diaria (ej. 0.0001 = 0.01%)
    - cash_threshold: proporción mínima de efectivo sobre el capital inicial para venta forzada
    """

    @dataclass
    class Operation:
        t: str
        p: float
        sl: float
        tp: float
        com: float
        n_shares: int
        type: str
        motivo: str

    # --- Detección automática de columnas ---
    def find_col(df, name):
        if name in df.columns:
            return name
        for col in df.columns:
            if isinstance(col, tuple) and name in col:
                return col
        raise KeyError(f"No se encontró la columna '{name}' en el DataFrame")

    col_close = find_col(df, 'Close')
    col_signal = find_col(df, 'signal')

    active_positions: list[Operation] = []
    records = []
    initial_cash = cash

    for i, row in df.iterrows():
        sig = row[col_signal]
        close_price = row[col_close]

        if isinstance(sig, (pd.Series, np.ndarray)):
            sig = sig.iloc[0] if hasattr(sig, 'iloc') else sig[0]
        if isinstance(close_price, (pd.Series, np.ndarray)):
            close_price = close_price.iloc[0] if hasattr(close_price, 'iloc') else close_price[0]

        # --- 1️⃣ Aplicar costo de financiamiento diario ---
        if borrow_rate > 0 and active_positions:
            for position in active_positions:
                cash -= position.n_shares * position.p * borrow_rate

        # --- 2️⃣ Cerrar posiciones por SL, TP o señal de venta ---
        for position in active_positions.copy():

            if close_price <= position.sl:
                cash += close_price * position.n_shares * (1 - com)
                active_positions.remove(position)
            elif sig == -1:
                cash += close_price * position.n_shares * (1 - com)
                active_positions.remove(position)

        # --- 3️⃣ Venta forzada si el cash es muy bajo ---
        if cash <= initial_cash * cash_threshold and active_positions:
            for position in active_positions.copy():
                cash += close_price * position.n_shares * (1 - com)
                active_positions.remove(position)

        # --- 4️⃣ Abrir posición si hay señal de compra ---
        if sig == 1:
            if n_shares == 0:
                n_to_buy = cash // (close_price * (1 + com))
            else:
                n_to_buy = n_shares  # usa el valor definido por el usuario

            cost = n_to_buy * close_price * (1 + com)
            if n_to_buy > 0 and cost <= cash:
                cash -= cost
                active_positions.append(Operation(
                    t=i,
                    p=close_price,
                    tp=close_price * (1 + tp),
                    sl=close_price * (1 - sl),
                    com=com,
                    n_shares=int(n_to_buy),
                    type="Long",
                    motivo="Compra señal"
                ))

        # --- 5️⃣ Calcular valor del portafolio ---
        invested_value = sum([pos.n_shares * close_price for pos in active_positions])
        total_portfolio = cash + invested_value
        rendimiento = (total_portfolio - initial_cash) / initial_cash

        records.append({
            'Date': i,
            'Cash': cash,
            'Portfolio': total_portfolio,
            'Rendimiento': rendimiento,
            'Active_Positions': len(active_positions)
        })

    return pd.DataFrame(records)
