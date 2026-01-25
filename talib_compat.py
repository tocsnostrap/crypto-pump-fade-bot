import numpy as np

try:
    import talib as _talib
    talib = _talib
    TALIB_SOURCE = "talib"
except Exception:
    import pandas_ta as ta
    TALIB_SOURCE = "pandas_ta"

    def _as_array(series, length):
        if series is None:
            return np.full(length, np.nan)
        arr = np.asarray(series)
        if arr.size == 0:
            return np.full(length, np.nan)
        if arr.size < length:
            pad = np.full(length - arr.size, np.nan)
            arr = np.concatenate([pad, arr])
        return arr

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

    talib = _TalibCompat()
