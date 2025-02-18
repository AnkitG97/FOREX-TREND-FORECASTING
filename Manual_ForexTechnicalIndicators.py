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
        # Trend Indicators - Moving Averages
        self.df['MA5'] = self.df['Close'].rolling(5).mean()
        self.df['MA10'] = self.df['Close'].rolling(10).mean()

    def _average_directional_index(self):
        """Corrected ADX implementation using pandas operations"""
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        
        # Calculate True Range using pandas
        tr = pd.DataFrame({
            'tr1': high - low,
            'tr2': (high - close.shift()).abs(),
            'tr3': (low - close.shift()).abs()
        }).max(axis=1)
        
        # Calculate Directional Movements using pandas
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=self.df.index)
        neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=self.df.index)
        
        # Smooth using Wilder's method (14-day)
        alpha = 1/14  # Wilder's smoothing factor
        
        # Use pandas' ewm instead of numpy
        tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
        pos_dm_s = pos_dm.ewm(alpha=alpha, adjust=False).mean()
        neg_dm_s = neg_dm.ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate Directional Indicators
        pos_di = 100 * (pos_dm_s / tr_s)
        neg_di = 100 * (neg_dm_s / tr_s)
        
        # Calculate ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        self.df['ADX'] = dx.ewm(alpha=alpha, adjust=False).mean()
        self.df['+DI14'] = pos_di
        self.df['-DI14'] = neg_di

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
        # Trend Indicators - MACD
        close = self.df['Close']
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        
        self.df['DIF'] = ema12 - ema26
        self.df['DEA'] = self.df['DIF'].ewm(span=9, adjust=False).mean()
        self.df['MACD'] = 2 * (self.df['DIF'] - self.df['DEA'])

    def _parabolic_sar(self):
        # Trend Indicators - Parabolic SAR
        high = self.df['High'].values
        low = self.df['Low'].values
        close = self.df['Close'].values
        
        sar = np.zeros(len(close))
        af = 0.02
        ep = 0
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        # Initial values
        sar[0] = low[0] if trend == 1 else high[0]
        
        for i in range(1, len(close)):
            if trend == 1:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                if low[i] < sar[i]:
                    trend = -1
                    sar[i] = ep
                    af = 0.02
                    ep = low[i]
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + 0.02, 0.2)
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                if high[i] > sar[i]:
                    trend = 1
                    sar[i] = ep
                    af = 0.02
                    ep = high[i]
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + 0.02, 0.2)
        
        self.df['PSAR'] = sar
        self.df['PSAR_TREND'] = np.where(self.df['PSAR'] < self.df['Close'], 1, -1)

    def _relative_strength_index(self):
        # Momentum Indicators - RSI
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 5-day RSI
        avg_gain5 = gain.rolling(5).mean()
        avg_loss5 = loss.rolling(5).mean()
        rs5 = avg_gain5 / avg_loss5
        self.df['RSI5'] = 100 - (100 / (1 + rs5))
        
        # 14-day RSI
        avg_gain14 = gain.rolling(14).mean()
        avg_loss14 = loss.rolling(14).mean()
        rs14 = avg_gain14 / avg_loss14
        self.df['RSI14'] = 100 - (100 / (1 + rs14))

    def _stochastic_oscillator(self):
        # Momentum Indicators - Stochastic
        high = self.df['High'].rolling(14).max()
        low = self.df['Low'].rolling(14).min()
        close = self.df['Close']
        
        self.df['%K'] = 100 * ((close - low) / (high - low))
        self.df['%D'] = self.df['%K'].rolling(3).mean()

    def _williams_percent_r(self):
        # Momentum Indicators - Williams %R
        high14 = self.df['High'].rolling(14).max()
        low14 = self.df['Low'].rolling(14).min()
        self.df['%R'] = -100 * ((high14 - self.df['Close']) / (high14 - low14))

    def _average_true_range(self):
        # Volatility Indicators - ATR (14-day)
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close'].shift()
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR14'] = tr.rolling(14).mean()

    def _bollinger_bands(self):
        # Volatility Indicators - Bollinger Bands
        ma20 = self.df['Close'].rolling(20).mean()
        std20 = self.df['Close'].rolling(20).std()
        
        self.df['BB_Middle'] = ma20
        self.df['BB_Upper'] = ma20 + 2 * std20
        self.df['BB_Lower'] = ma20 - 2 * std20

    def _sample_std_dev(self):
        # Volatility Indicators - Sample Std Dev
        self.df['STD20'] = self.df['Close'].rolling(20).std(ddof=1)
