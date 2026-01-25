import numpy as np


def _as_array(series, length):
    if series is None:
        return np.full(length, np.nan)
    arr = np.asarray(series, dtype="float64")
    if arr.size == 0:
        return np.full(length, np.nan)
    if arr.size < length:
        pad = np.full(length - arr.size, np.nan)
        arr = np.concatenate([pad, arr])
    return arr


try:
    import talib as _talib
    talib = _talib
    TALIB_SOURCE = "talib"
except Exception:
    try:
        import pandas_ta as ta

        TALIB_SOURCE = "pandas_ta"

        class _TalibCompat:
            @staticmethod
            def RSI(closes, timeperiod=14):
                series = ta.rsi(closes, length=timeperiod)
                return _as_array(series, len(closes))

            @staticmethod
            def MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9):
                df = ta.macd(closes, fast=fastperiod, slow=slowperiod, signal=signalperiod)
                length = len(closes)
                if df is None or df.empty:
                    empty = np.full(length, np.nan)
                    return empty, empty, empty
                macd_col = f"MACD_{fastperiod}_{slowperiod}_{signalperiod}"
                signal_col = f"MACDs_{fastperiod}_{slowperiod}_{signalperiod}"
                hist_col = f"MACDh_{fastperiod}_{slowperiod}_{signalperiod}"
                return (
                    _as_array(df[macd_col], length),
                    _as_array(df[signal_col], length),
                    _as_array(df[hist_col], length),
                )

            @staticmethod
            def BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
                df = ta.bbands(closes, length=timeperiod, std=nbdevup)
                length = len(closes)
                if df is None or df.empty:
                    empty = np.full(length, np.nan)
                    return empty, empty, empty
                suffix = f"{float(nbdevup)}"
                upper_col = f"BBU_{timeperiod}_{suffix}"
                middle_col = f"BBM_{timeperiod}_{suffix}"
                lower_col = f"BBL_{timeperiod}_{suffix}"
                return (
                    _as_array(df[upper_col], length),
                    _as_array(df[middle_col], length),
                    _as_array(df[lower_col], length),
                )

            @staticmethod
            def ATR(highs, lows, closes, timeperiod=14):
                series = ta.atr(high=highs, low=lows, close=closes, length=timeperiod)
                return _as_array(series, len(closes))

            @staticmethod
            def SMA(values, timeperiod=20):
                series = ta.sma(values, length=timeperiod)
                return _as_array(series, len(values))

            @staticmethod
            def EMA(values, timeperiod=20):
                series = ta.ema(values, length=timeperiod)
                return _as_array(series, len(values))

        talib = _TalibCompat()
    except Exception:
        import pandas as pd

        TALIB_SOURCE = "compat"

        def _series(values):
            return pd.Series(values, dtype="float64")

        class _TalibCompat:
            @staticmethod
            def RSI(closes, timeperiod=14):
                series = _series(closes)
                delta = series.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(alpha=1 / timeperiod, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1 / timeperiod, adjust=False).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return _as_array(rsi, len(closes))

            @staticmethod
            def MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9):
                series = _series(closes)
                ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
                ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
                macd = ema_fast - ema_slow
                signal = macd.ewm(span=signalperiod, adjust=False).mean()
                hist = macd - signal
                length = len(closes)
                return (
                    _as_array(macd, length),
                    _as_array(signal, length),
                    _as_array(hist, length),
                )

            @staticmethod
            def BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
                series = _series(closes)
                middle = series.rolling(window=timeperiod).mean()
                std = series.rolling(window=timeperiod).std(ddof=0)
                upper = middle + (std * nbdevup)
                lower = middle - (std * nbdevdn)
                length = len(closes)
                return (
                    _as_array(upper, length),
                    _as_array(middle, length),
                    _as_array(lower, length),
                )

            @staticmethod
            def ATR(highs, lows, closes, timeperiod=14):
                high = _series(highs)
                low = _series(lows)
                close = _series(closes)
                prev_close = close.shift(1)
                tr_components = pd.concat(
                    [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                    axis=1,
                )
                tr = tr_components.max(axis=1)
                atr = tr.ewm(alpha=1 / timeperiod, adjust=False).mean()
                return _as_array(atr, len(closes))

            @staticmethod
            def SMA(values, timeperiod=20):
                series = _series(values).rolling(window=timeperiod).mean()
                return _as_array(series, len(values))

            @staticmethod
            def EMA(values, timeperiod=20):
                series = _series(values).ewm(span=timeperiod, adjust=False).mean()
                return _as_array(series, len(values))

        talib = _TalibCompat()
