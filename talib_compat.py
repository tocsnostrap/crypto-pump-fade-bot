import numpy as np

try:
    import talib as _talib
    talib = _talib
    TALIB_SOURCE = "talib"
except Exception:
    try:
        import pandas_ta as ta
        TALIB_SOURCE = "pandas_ta"
    except ImportError:
        # Fallback to 'ta' library if pandas-ta fails
        import ta
        from ta.momentum import RSIIndicator
        from ta.trend import MACD, SMAIndicator
        from ta.volatility import BollingerBands, AverageTrueRange
        import pandas as pd
        TALIB_SOURCE = "ta"

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
            if TALIB_SOURCE == "pandas_ta":
                series = ta.rsi(closes, length=timeperiod)
            else:
                # 'ta' library implementation
                series = RSIIndicator(close=pd.Series(closes), window=timeperiod).rsi()
            return _as_array(series, len(closes))

        @staticmethod
        def MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9):
            length = len(closes)
            if TALIB_SOURCE == "pandas_ta":
                df = ta.macd(closes, fast=fastperiod, slow=slowperiod, signal=signalperiod)
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
            else:
                # 'ta' library implementation
                indicator = MACD(close=pd.Series(closes), window_slow=slowperiod, window_fast=fastperiod, window_sign=signalperiod)
                return (
                    _as_array(indicator.macd(), length),
                    _as_array(indicator.macd_signal(), length),
                    _as_array(indicator.macd_diff(), length)
                )

        @staticmethod
        def BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
            length = len(closes)
            if TALIB_SOURCE == "pandas_ta":
                df = ta.bbands(closes, length=timeperiod, std=nbdevup)
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
            else:
                # 'ta' library implementation
                indicator = BollingerBands(close=pd.Series(closes), window=timeperiod, window_dev=nbdevup)
                return (
                    _as_array(indicator.bollinger_hband(), length),
                    _as_array(indicator.bollinger_mavg(), length),
                    _as_array(indicator.bollinger_lband(), length)
                )

        @staticmethod
        def ATR(highs, lows, closes, timeperiod=14):
            if TALIB_SOURCE == "pandas_ta":
                series = ta.atr(high=highs, low=lows, close=closes, length=timeperiod)
            else:
                # 'ta' library implementation
                indicator = AverageTrueRange(high=pd.Series(highs), low=pd.Series(lows), close=pd.Series(closes), window=timeperiod)
                series = indicator.average_true_range()
            return _as_array(series, len(closes))

        @staticmethod
        def SMA(values, timeperiod=20):
            if TALIB_SOURCE == "pandas_ta":
                series = ta.sma(values, length=timeperiod)
            else:
                # 'ta' library implementation
                indicator = SMAIndicator(close=pd.Series(values), window=timeperiod)
                series = indicator.sma_indicator()
            return _as_array(series, len(values))

    talib = _TalibCompat()
