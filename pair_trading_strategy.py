"""
Pair Trading Strategy Using Polynomial Regression
Self Project (May '25 - June '25)

- Market-neutral pairs strategy: model the relationship between two
  cointegrated stocks with polynomial regression (rather than a plain
  linear hedge ratio) to capture non-linear mean-reversion in the spread.
- Statistical validation with the Augmented Dickey-Fuller (ADF) test and
  the Engle-Granger two-step cointegration test before trading a pair.
- Backtested end-to-end in Python using Backtrader / Cerebro on historical
  daily equity data, reporting returns, Sharpe ratio, and drawdown.

Usage:
    python pair_trading_strategy.py --tickers KO PEP --start 2018-01-01 --end 2024-01-01

Dependencies:
    pip install yfinance backtrader statsmodels numpy pandas matplotlib --break-system-packages
"""

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Downloads adjusted close prices for the given tickers via yfinance."""
    import yfinance as yf
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data = data.dropna()
    return data


# --------------------------------------------------------------------------
# 2. Cointegration testing: ADF + Engle-Granger two-step method
# --------------------------------------------------------------------------
@dataclass
class CointegrationResult:
    is_cointegrated: bool
    adf_pvalue: float
    eg_pvalue: float
    hedge_ratio: float
    intercept: float


def test_cointegration(y: pd.Series, x: pd.Series, significance: float = 0.05) -> CointegrationResult:
    """
    Engle-Granger two-step test:
      1. Regress y on x (OLS) to get the equilibrium relationship:
             y_t = alpha + beta * x_t + spread_t
      2. Run an Augmented Dickey-Fuller test on the residual "spread" series.
         If the spread is stationary (ADF rejects the unit-root null), the
         two series are cointegrated and a mean-reverting spread exists to
         trade.
    We also run statsmodels' built-in `coint` test as a cross-check.
    """
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.api import OLS, add_constant

    x_const = add_constant(x)
    ols_result = OLS(y, x_const).fit()
    hedge_ratio = ols_result.params.iloc[1]
    intercept = ols_result.params.iloc[0]
    spread = y - (hedge_ratio * x + intercept)

    adf_pvalue = adfuller(spread, autolag="AIC")[1]
    eg_pvalue = coint(y, x)[1]

    is_cointegrated = (adf_pvalue < significance) and (eg_pvalue < significance)
    return CointegrationResult(is_cointegrated, adf_pvalue, eg_pvalue, hedge_ratio, intercept)


# --------------------------------------------------------------------------
# 3. Polynomial regression hedge model
#
#    Instead of a fixed linear hedge ratio, fit y_t ~ f(x_t) where f is a
#    degree-d polynomial. This lets the model capture curvature in the
#    price relationship (e.g. relative valuation effects that aren't
#    perfectly linear) while the *residual* (spread) is still what we test
#    for stationarity and trade on.
# --------------------------------------------------------------------------
class PolynomialSpreadModel:
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.coeffs: np.ndarray | None = None

    def fit(self, x: pd.Series, y: pd.Series):
        self.coeffs = np.polyfit(x.values, y.values, deg=self.degree)
        return self

    def predict(self, x: pd.Series) -> np.ndarray:
        return np.polyval(self.coeffs, x.values)

    def spread(self, x: pd.Series, y: pd.Series) -> pd.Series:
        fitted = self.predict(x)
        return pd.Series(y.values - fitted, index=y.index, name="spread")


def zscore(series: pd.Series, window: int) -> pd.Series:
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    return (series - roll_mean) / roll_std


# --------------------------------------------------------------------------
# 4. Backtrader strategy: trade the polynomial-regression spread's z-score
# --------------------------------------------------------------------------
import backtrader as bt


class PolyRegPairsStrategy(bt.Strategy):
    params = dict(
        z_window=30,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,      # hard stop-loss if the spread diverges further
        poly_degree=2,
        alloc=0.45,      # fraction of portfolio value per leg
    )

    def __init__(self):
        self.y = self.datas[0].close   # dependent asset
        self.x = self.datas[1].close   # independent asset
        self.spread_history: list[float] = []
        self.model = PolynomialSpreadModel(degree=self.p.poly_degree)
        self.order_pending = False

    def log(self, msg):
        dt = self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} | {msg}")

    def next(self):
        if len(self) < self.p.z_window + 5 or self.order_pending:
            return

        # Refit the polynomial hedge model on a trailing window (rolling fit
        # keeps the hedge ratio adaptive rather than static for the whole backtest).
        y_win = pd.Series([self.y[-i] for i in range(self.p.z_window, 0, -1)])
        x_win = pd.Series([self.x[-i] for i in range(self.p.z_window, 0, -1)])
        self.model.fit(x_win, y_win)

        fitted_now = self.model.predict(pd.Series([self.x[0]]))[0]
        current_spread = self.y[0] - fitted_now
        self.spread_history.append(current_spread)

        if len(self.spread_history) < self.p.z_window:
            return

        recent = np.array(self.spread_history[-self.p.z_window:])
        z = (current_spread - recent.mean()) / (recent.std() + 1e-9)

        pos_y = self.getposition(self.datas[0]).size
        pos_x = self.getposition(self.datas[1]).size
        in_position = pos_y != 0 or pos_x != 0

        cash = self.broker.get_value()
        size_y = int((cash * self.p.alloc) / self.y[0])
        size_x = int((cash * self.p.alloc) / self.x[0])

        if not in_position:
            if z > self.p.entry_z:
                # Spread too high -> short y, long x (bet on reversion down).
                self.sell(data=self.datas[0], size=size_y)
                self.buy(data=self.datas[1], size=size_x)
                self.log(f"ENTRY short-spread | z={z:.2f}")
            elif z < -self.p.entry_z:
                # Spread too low -> long y, short x (bet on reversion up).
                self.buy(data=self.datas[0], size=size_y)
                self.sell(data=self.datas[1], size=size_x)
                self.log(f"ENTRY long-spread | z={z:.2f}")
        else:
            exit_signal = abs(z) < self.p.exit_z
            stop_signal = abs(z) > self.p.stop_z
            if exit_signal or stop_signal:
                self.close(data=self.datas[0])
                self.close(data=self.datas[1])
                reason = "stop-loss" if stop_signal else "mean-reversion exit"
                self.log(f"EXIT ({reason}) | z={z:.2f}")


# --------------------------------------------------------------------------
# 5. Backtest runner (Backtrader / Cerebro)
# --------------------------------------------------------------------------
def run_backtest(prices: pd.DataFrame, ticker_y: str, ticker_x: str, cash: float = 100_000.0):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)  # 10 bps per trade, a realistic default

    for ticker in (ticker_y, ticker_x):
        feed = bt.feeds.PandasData(
            dataname=pd.DataFrame({
                "open": prices[ticker], "high": prices[ticker],
                "low": prices[ticker], "close": prices[ticker],
                "volume": 0,
            }, index=prices.index)
        )
        cerebro.adddata(feed, name=ticker)

    cerebro.addstrategy(PolyRegPairsStrategy)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    strat = results[0]

    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    max_dd = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown")
    total_return = (end_value / start_value - 1) * 100

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Start portfolio value : ${start_value:,.2f}")
    print(f"End portfolio value   : ${end_value:,.2f}")
    print(f"Total return          : {total_return:.2f}%")
    print(f"Sharpe ratio          : {sharpe:.3f}" if sharpe else "Sharpe ratio          : n/a")
    print(f"Max drawdown          : {max_dd:.2f}%" if max_dd else "Max drawdown          : n/a")
    return cerebro


# --------------------------------------------------------------------------
# 6. Main pipeline: load data -> test cointegration -> backtest
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Polynomial-regression pairs trading strategy")
    parser.add_argument("--tickers", nargs=2, default=["KO", "PEP"], metavar=("Y", "X"))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()

    ticker_y, ticker_x = args.tickers
    print(f"Loading price data for {ticker_y}, {ticker_x} ...")
    prices = load_price_data([ticker_y, ticker_x], args.start, args.end)

    print("\nRunning cointegration tests (ADF + Engle-Granger)...")
    coint_result = test_cointegration(prices[ticker_y], prices[ticker_x])
    print(f"  Hedge ratio (linear OLS): {coint_result.hedge_ratio:.4f}")
    print(f"  ADF p-value             : {coint_result.adf_pvalue:.4f}")
    print(f"  Engle-Granger p-value   : {coint_result.eg_pvalue:.4f}")
    print(f"  Cointegrated (5% level) : {coint_result.is_cointegrated}")

    if not coint_result.is_cointegrated:
        print("\nWarning: pair does not show strong statistical cointegration. "
              "Proceeding with backtest for demonstration, but treat results with caution.")

    print(f"\nFitting degree-{args.degree} polynomial spread model on full sample for reference...")
    model = PolynomialSpreadModel(degree=args.degree).fit(prices[ticker_x], prices[ticker_y])
    spread = model.spread(prices[ticker_x], prices[ticker_y])
    print(f"  Spread mean: {spread.mean():.4f}, std: {spread.std():.4f}")

    print("\nRunning Backtrader backtest...")
    run_backtest(prices, ticker_y, ticker_x)


if __name__ == "__main__":
    main()
