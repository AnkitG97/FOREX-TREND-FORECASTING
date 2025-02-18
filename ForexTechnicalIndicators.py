import talib
import pandas as pd
import numpy as np

class ForexTechnicalIndicators:
    def __init__(self, ohlc_df):
        self.df = ohlc_df.copy()
        self._validate_columns()
        
    def _validate_columns(self):
        required = ['Open', 'High', 'Low', 'Close']
        if not all(col in self.df.columns for col in required):
            raise ValueError("DataFrame must contain OHLC columns")

    def calculate_all_indicators(self):
        # Trend Indicators
        self._moving_averages()
        self._average_directional_index()
        self._ichimoku_cloud()
        self._macd()
        self._parabolic_sar()
        
        # Momentum Indicators
        self._relative_strength_index()
        self._stochastic_oscillator()
        self._williams_percent_r()
        
        # Volatility Indicators
        self._average_true_range()
        self._bollinger_bands()
        self._sample_std_dev()
        
        return self.df

    def _moving_averages(self):
        # Simple Moving Averages using TA-Lib
        self.df['MA5'] = talib.SMA(self.df['Close'], timeperiod=5)
        self.df['MA10'] = talib.SMA(self.df['Close'], timeperiod=10)

    def _average_directional_index(self):
        # ADX and Directional Indicators using TA-Lib
        self.df['ADX'] = talib.ADX(self.df['High'], self.df['Low'], self.df['Close'], timeperiod=14)
        self.df['+DI14'] = talib.PLUS_DI(self.df['High'], self.df['Low'], self.df['Close'], timeperiod=14)
        self.df['-DI14'] = talib.MINUS_DI(self.df['High'], self.df['Low'], self.df['Close'], timeperiod=14)

    def _ichimoku_cloud(self):
        # Trend Indicators - Ichimoku (7, 22, 44)
        high = self.df['High']
        low = self.df['Low']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(7).max()
        tenkan_low = low.rolling(7).min()
        self.df['Tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(22).max()
        kijun_low = low.rolling(22).min()
        self.df['Kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        self.df['Senkou_A'] = ((self.df['Tenkan_sen'] + self.df['Kijun_sen']) / 2).shift(22)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(44).max()
        senkou_low = low.rolling(44).min()
        self.df['Senkou_B'] = ((senkou_high + senkou_low) / 2).shift(22)

    def _macd(self):
        # MACD components using TA-Lib
        macd, signal, hist = talib.MACD(
            self.df['Close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        self.df['DIF'] = macd
        self.df['DEA'] = signal
        self.df['MACD'] = 2 * hist  # Scale histogram to match original implementation

    def _parabolic_sar(self):
        # Parabolic SAR using TA-Lib with standard acceleration
        self.df['PSAR'] = talib.SAR(self.df['High'], self.df['Low'], acceleration=0.02, maximum=0.2)
        self.df['PSAR_TREND'] = np.where(self.df['PSAR'] < self.df['Close'], 1, -1)

    def _relative_strength_index(self):
        # RSI using TA-Lib for both 5 and 14 periods
        self.df['RSI5'] = talib.RSI(self.df['Close'], timeperiod=5)
        self.df['RSI14'] = talib.RSI(self.df['Close'], timeperiod=14)

    def _stochastic_oscillator(self):
        # Stochastic Oscillator using TA-Lib
        slowk, slowd = talib.STOCH(
            self.df['High'], self.df['Low'], self.df['Close'],
            fastk_period=14,
            slowk_period=1,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        self.df['%K'] = slowk
        self.df['%D'] = slowd

    def _williams_percent_r(self):
        # Williams %R using TA-Lib
        self.df['%R'] = talib.WILLR(self.df['High'], self.df['Low'], self.df['Close'], timeperiod=14)

    def _average_true_range(self):
        # Average True Range using TA-Lib
        self.df['ATR14'] = talib.ATR(self.df['High'], self.df['Low'], self.df['Close'], timeperiod=14)

    def _bollinger_bands(self):
        # Bollinger Bands using TA-Lib with 20-period SMA
        upper, middle, lower = talib.BBANDS(
            self.df['Close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        self.df['BB_Middle'] = middle
        self.df['BB_Upper'] = upper
        self.df['BB_Lower'] = lower

    def _sample_std_dev(self):
        # Standard Deviation using TA-Lib (sample std dev with ddof=1)
        self.df['STD20'] = talib.STDDEV(self.df['Close'], timeperiod=20, nbdev=1)