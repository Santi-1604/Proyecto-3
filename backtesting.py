import pandas as pd
import numpy as np
from dataclasses import dataclass

def backtest(df, cash, com, n_shares, sl, tp,
             borrow_rate=0.0, cash_threshold=0.01):
    """
    Backtest con métricas de rentabilidad y riesgo.

    Parámetros:
    - df: DataFrame con columnas ['Date', 'Close', 'signal']
    - cash: capital inicial
    - com: comisión (por operación)
    - n_shares: número fijo de acciones (0 = usar todo el capital)
    - sl: stop loss (proporcional, ej. 0.05)
    - tp: take profit (proporcional, ej. 0.05)
    - borrow_rate: tasa de financiamiento diaria
    - cash_threshold: proporción mínima de efectivo antes de vender forzadamente

    Retorna:
    - df_bt: DataFrame con la evolución del portafolio
    - metrics: diccionario con métricas del backtest
    """

    @dataclass
    class Operation:
        t: pd.Timestamp
        p: float
        sl: float
        tp: float
        com: float
        n_shares: int
        type: str
        motivo: str

    # === Función de tasa libre de riesgo por año ===
    def risk_free_rate(date):
        """Devuelve una tasa libre de riesgo anual estimada según el año."""
        year = pd.to_datetime(date).year
        rf_table = {
            2020: 0.002,
            2021: 0.005,
            2022: 0.043,
            2023: 0.0541,
            2024: 0.04,
            2025: 0.03604
        }
        return rf_table.get(year, 0.03)  # default 3%

    # --- Detectar columnas ---
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
    closed_trades = []  # registro de trades cerrados
    records = []
    initial_cash = cash

    for i, row in df.iterrows():
        date = pd.to_datetime(i)
        sig = row[col_signal]
        close_price = row[col_close]

        # --- Corrección de formato ---
        if isinstance(sig, (pd.Series, np.ndarray)):
            sig = sig.iloc[0] if hasattr(sig, 'iloc') else sig[0]
        if isinstance(close_price, (pd.Series, np.ndarray)):
            close_price = close_price.iloc[0] if hasattr(close_price, 'iloc') else close_price[0]

        # --- 1️⃣ Costo de financiamiento diario ---
        if borrow_rate > 0 and active_positions:
            for position in active_positions:
                cash -= position.n_shares * position.p * borrow_rate

        # --- 2️⃣ Cierre de posiciones ---
        for position in active_positions.copy():
            # Stop Loss
            if close_price <= position.sl:
                cash += close_price * position.n_shares * (1 - com)
                pnl = (close_price - position.p) / position.p
                closed_trades.append({'Open_Date': position.t, 'Close_Date': date, 'PnL': pnl})
                active_positions.remove(position)

            # Take Profit
            elif close_price >= position.tp:
                cash += close_price * position.n_shares * (1 - com)
                pnl = (close_price - position.p) / position.p
                closed_trades.append({'Open_Date': position.t, 'Close_Date': date, 'PnL': pnl})
                active_positions.remove(position)

            # Señal de venta
            elif sig == -1:
                cash += close_price * position.n_shares * (1 - com)
                pnl = (close_price - position.p) / position.p
                closed_trades.append({'Open_Date': position.t, 'Close_Date': date, 'PnL': pnl})
                active_positions.remove(position)

        # --- 3️⃣ Venta forzada ---
        if cash <= initial_cash * cash_threshold and active_positions:
            for position in active_positions.copy():
                cash += close_price * position.n_shares * (1 - com)
                pnl = (close_price - position.p) / position.p
                closed_trades.append({'Open_Date': position.t, 'Close_Date': date, 'PnL': pnl})
                active_positions.remove(position)

        # --- 4️⃣ Apertura de posición ---
        if sig == 1:
            if n_shares == 0:
                n_to_buy = cash // (close_price * (1 + com))
            else:
                n_to_buy = n_shares

            cost = n_to_buy * close_price * (1 + com)
            if n_to_buy > 0 and cost <= cash:
                cash -= cost
                active_positions.append(Operation(
                    t=date,
                    p=close_price,
                    tp=close_price * (1 + tp),
                    sl=close_price * (1 - sl),
                    com=com,
                    n_shares=int(n_to_buy),
                    type="Long",
                    motivo="Compra señal"
                ))

        # --- 5️⃣ Valor del portafolio ---
        invested_value = sum([pos.n_shares * close_price for pos in active_positions])
        total_portfolio = cash + invested_value
        rendimiento = (total_portfolio - initial_cash) / initial_cash

        records.append({
            'Date': date,
            'Cash': cash,
            'Portfolio': total_portfolio,
            'Rendimiento': rendimiento,
            'Active_Positions': len(active_positions)
        })

    df_bt = pd.DataFrame(records).sort_values('Date').reset_index(drop=True)

    # === 6️⃣ Métricas de desempeño ===
    df_bt['Returns'] = df_bt['Portfolio'].pct_change().fillna(0)
    df_bt['Drawdown'] = df_bt['Portfolio'] / df_bt['Portfolio'].cummax() - 1
    max_drawdown = df_bt['Drawdown'].min()

    # --- Tasa libre de riesgo promedio diaria ---
    df_bt['rf'] = df_bt['Date'].apply(risk_free_rate) / 252
    excess_return = df_bt['Returns'] - df_bt['rf']

    sharpe_ratio = np.sqrt(252) * excess_return.mean() / (excess_return.std() + 1e-9)
    downside_returns = df_bt.loc[df_bt['Returns'] < 0, 'Returns']
    sortino_ratio = np.sqrt(252) * excess_return.mean() / (downside_returns.std() + 1e-9)
    calmar_ratio = (df_bt['Portfolio'].iloc[-1] / df_bt['Portfolio'].iloc[0] - 1) / abs(max_drawdown)

    # --- Métricas de operaciones ---
    df_trades = pd.DataFrame(closed_trades)
    total_trades = len(df_trades)
    win_rate = np.nan
    if total_trades > 0:
        wins = (df_trades['PnL'] > 0).sum()
        win_rate = wins / total_trades

    metrics = {
        'Final Portfolio': df_bt['Portfolio'].iloc[-1],
        'Total Return': df_bt['Rendimiento'].iloc[-1],
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown,
        'Annualized Volatility': df_bt['Returns'].std() * np.sqrt(252),
        'Total Trades': total_trades,
        'Win Rate': win_rate
    }

    return df_bt, metrics
