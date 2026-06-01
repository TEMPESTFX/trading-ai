import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class StrategyAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def detect_trend_24h(df_1d: pd.DataFrame) -> str:
        """
        Détermine la tendance globale sur l'unité de temps 24h (1 Jour).
        Retourne 'BULLISH', 'BEARISH' ou 'NEUTRAL'.
        """
        if len(df_1d) < 50:
            return "NEUTRAL"
        
        # Utilisation de l'EMA 50 et EMA 20
        ema_20 = df_1d['close'].ewm(span=20, adjust=False).mean()
        ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
        
        latest_ema_20 = ema_20.iloc[-1]
        latest_ema_50 = ema_50.iloc[-1]
        
        # Pente de l'EMA 50 sur les 3 dernières bougies
        ema_50_slope = (ema_50.iloc[-1] - ema_50.iloc[-3]) / 3
        
        if latest_ema_20 > latest_ema_50 and ema_50_slope > 0:
            return "BULLISH"
        elif latest_ema_20 < latest_ema_50 and ema_50_slope < 0:
            return "BEARISH"
        else:
            return "NEUTRAL"

    @staticmethod
    def find_support_resistance(df_4h: pd.DataFrame, df_2h: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identifie les niveaux de Support et Résistance clés sur les timeframes 4h et 2h.
        """
        levels = {"supports": [], "resistances": []}
        
        for df in [df_4h, df_2h]:
            if len(df) < 20:
                continue
            
            # Recherche des extrema locaux sur une fenêtre de 5 bougies (gauche/droite)
            window = 5
            for i in range(window, len(df) - window):
                low_val = df['low'].iloc[i]
                high_val = df['high'].iloc[i]
                
                # Support (creux local)
                if low_val == df['low'].iloc[i-window:i+window+1].min():
                    levels["supports"].append(float(low_val))
                
                # Résistance (pic local)
                if high_val == df['high'].iloc[i-window:i+window+1].max():
                    levels["resistances"].append(float(high_val))
                    
        # Nettoyage et regroupement des niveaux proches (clusterisation simplifiée)
        levels["supports"] = StrategyAnalyzer._cluster_levels(levels["supports"])
        levels["resistances"] = StrategyAnalyzer._cluster_levels(levels["resistances"])
        
        return levels

    @staticmethod
    def _cluster_levels(levels: List[float], threshold_pct: float = 0.005) -> List[float]:
        """Regroupe les niveaux de prix proches pour éviter les doublons."""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for price in levels[1:]:
            mean_val = np.mean(current_cluster)
            # Si le prix est à moins de threshold_pct% du cluster actuel
            if abs(price - mean_val) / mean_val < threshold_pct:
                current_cluster.append(price)
            else:
                clustered.append(float(np.mean(current_cluster)))
                current_cluster = [price]
                
        clustered.append(float(np.mean(current_cluster)))
        return clustered

    @staticmethod
    def detect_bos_retest(df_lh: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """
        Détecte les Cassures de Structure (Break of Structure - BoS) et signale un Retest.
        Retourne ('BUY', trigger_price) ou ('SELL', trigger_price) ou (None, None).
        """
        if len(df_lh) < 15:
            return None, None
            
        highs = df_lh['high'].values
        lows = df_lh['low'].values
        closes = df_lh['close'].values
        
        # Trouver les derniers swings high et low
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df_lh) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append((i, lows[i]))
                
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None, None
            
        last_sh_idx, last_sh_val = swing_highs[-1]
        last_sl_idx, last_sl_val = swing_lows[-1]
        
        current_price = closes[-1]
        prev_price = closes[-2]
        
        # 1. Scénario Bullish BoS & Retest:
        # Le prix actuel casse la dernière résistance majeure (swing high), puis y revient pour retester.
        # Cassure : Un prix récent a clôturé au-dessus du dernier swing high.
        recent_breakout_buy = any(closes[last_sh_idx:-1] > last_sh_val)
        # Retest actuel : Le prix bas récent touche ou s'approche de l'ancien swing high par le haut
        if recent_breakout_buy and current_price > last_sh_val:
            # Retest de la zone (marge de 0.1%)
            if abs(df_lh['low'].iloc[-2:] - last_sh_val).min() / last_sh_val < 0.0015:
                # Bougie de rejet haussière (mèche basse ou verte)
                if current_price > df_lh['open'].iloc[-1]:
                    return "BUY", last_sh_val
                    
        # 2. Scénario Bearish BoS & Retest:
        # Le prix actuel casse le dernier support majeur (swing low), puis y revient pour retester.
        recent_breakout_sell = any(closes[last_sl_idx:-1] < last_sl_val)
        if recent_breakout_sell and current_price < last_sl_val:
            # Retest de la zone (marge de 0.1%)
            if abs(df_lh['high'].iloc[-2:] - last_sl_val).min() / last_sl_val < 0.0015:
                # Bougie de rejet baissière (mèche haute ou rouge)
                if current_price < df_lh['open'].iloc[-1]:
                    return "SELL", last_sl_val
                    
        return None, None

    @staticmethod
    def detect_fibonacci_retracement(df_lh: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """
        Détecte une impulsion majeure et calcule les retracements de Fibonacci (0.5 et 0.618).
        Retourne ('BUY', fib_entry_price) ou ('SELL', fib_entry_price) ou (None, None).
        """
        if len(df_lh) < 20:
            return None, None
            
        closes = df_lh['close'].values
        highs = df_lh['high'].values
        lows = df_lh['low'].values
        
        # Identifier le plus bas et le plus haut majeurs des 20 dernières bougies
        idx_min = np.argmin(lows[-20:]) + len(df_lh) - 20
        idx_max = np.argmax(highs[-20:]) + len(df_lh) - 20
        
        price_min = lows[idx_min]
        price_max = highs[idx_max]
        
        diff = price_max - price_min
        if diff <= 0:
            return None, None
            
        current_price = closes[-1]
        
        # Impulsion Haussière (le minimum vient avant le maximum)
        if idx_min < idx_max:
            # Calcul des niveaux de retracement
            fib_50 = price_max - 0.5 * diff
            fib_618 = price_max - 0.618 * diff
            
            # Si le prix actuel est dans la zone dorée [fib_618, fib_50]
            if fib_618 <= current_price <= fib_50:
                # Indication de rejet haussier (ex: bougie actuelle verte ou mèche basse importante)
                if current_price > df_lh['open'].iloc[-1]:
                    return "BUY", fib_618
                    
        # Impulsion Baissière (le maximum vient avant le minimum)
        elif idx_max < idx_min:
            fib_50 = price_min + 0.5 * diff
            fib_618 = price_min + 0.618 * diff
            
            # Si le prix actuel est dans la zone dorée [fib_50, fib_618]
            if fib_50 <= current_price <= fib_618:
                # Indication de rejet baissier (ex: bougie actuelle rouge)
                if current_price < df_lh['open'].iloc[-1]:
                    return "SELL", fib_618
                    
        return None, None

    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> Tuple[Optional[str], str]:
        """
        Détecte les figures chartistes basiques comme le Double Top et le Double Bottom.
        Retourne ('BUY' ou 'SELL' ou None, 'Nom de la figure').
        """
        if len(df) < 30:
            return None, "Aucune"
            
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Trouver les sommets locaux (peaks) et creux locaux (valleys)
        peaks = []
        valleys = []
        for i in range(2, len(df) - 2):
            if highs[i] == max(highs[i-2:i+3]):
                peaks.append((i, highs[i]))
            if lows[i] == min(lows[i-2:i+3]):
                valleys.append((i, lows[i]))
                
        # 1. Détection Double Bottom (Achat potentiel)
        # Nécessite au moins 2 creux à un niveau similaire avec un sommet entre les deux
        if len(valleys) >= 2 and len(peaks) >= 1:
            v1_idx, v1_val = valleys[-2]
            v2_idx, v2_val = valleys[-1]
            
            # Trouver le sommet le plus haut entre les deux creux
            mid_peaks = [p for p in peaks if v1_idx < p[0] < v2_idx]
            if mid_peaks:
                peak_idx, peak_val = max(mid_peaks, key=lambda x: x[1])
                
                # Condition 1: Les deux creux sont à un niveau similaire (différence < 0.2%)
                diff_valleys = abs(v1_val - v2_val) / max(v1_val, v2_val)
                # Condition 2: Cassure de la ligne de cou (neckline = peak_val)
                if diff_valleys < 0.002 and closes[-1] > peak_val:
                    return "BUY", "Double Bottom"
                    
        # 2. Détection Double Top (Vente potentielle)
        # Nécessite au moins 2 sommets à un niveau similaire avec un creux entre les deux
        if len(peaks) >= 2 and len(valleys) >= 1:
            p1_idx, p1_val = peaks[-2]
            p2_idx, p2_val = peaks[-1]
            
            # Trouver le creux le plus bas entre les deux sommets
            mid_valleys = [v for v in valleys if p1_idx < v[0] < p2_idx]
            if mid_valleys:
                valley_idx, valley_val = min(mid_valleys, key=lambda x: x[1])
                
                # Condition 1: Les deux sommets sont à un niveau similaire (différence < 0.2%)
                diff_peaks = abs(p1_val - p2_val) / max(p1_val, p2_val)
                # Condition 2: Cassure de la ligne de cou (neckline = valley_val)
                if diff_peaks < 0.002 and closes[-1] < valley_val:
                    return "SELL", "Double Top"
                    
        return None, "Aucune"

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return (macd - signal_line).fillna(0.0)

    @staticmethod
    def compute_momentum(series: pd.Series, period: int = 8) -> pd.Series:
        return series.diff(period).fillna(0.0)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < 2:
            return 0.0
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean().iloc[-1]
        return max(float(atr), float(df['high'].iloc[-1] - df['low'].iloc[-1]))

    def is_in_buy_order_block(self, df_4h: pd.DataFrame, df_2h: pd.DataFrame, df_exec: pd.DataFrame) -> bool:
        levels = self.find_support_resistance(df_4h, df_2h)
        if not levels["supports"]:
            return False
        current_price = df_exec['close'].iloc[-1]
        support = levels["supports"][-1]
        if current_price - support > support * 0.006:
            return False
        last = df_exec.iloc[-1]
        prev = df_exec.iloc[-2]
        return last['close'] > last['open'] and prev['close'] < prev['open'] and last['low'] <= support * 1.0015

    def is_in_sell_order_block(self, df_4h: pd.DataFrame, df_2h: pd.DataFrame, df_exec: pd.DataFrame) -> bool:
        levels = self.find_support_resistance(df_4h, df_2h)
        if not levels["resistances"]:
            return False
        current_price = df_exec['close'].iloc[-1]
        resistance = levels["resistances"][-1]
        if resistance - current_price > resistance * 0.006:
            return False
        last = df_exec.iloc[-1]
        prev = df_exec.iloc[-2]
        return last['close'] < last['open'] and prev['close'] > prev['open'] and last['high'] >= resistance * 0.9985

    def generate_signal(self, df_1d: pd.DataFrame, df_4h: pd.DataFrame, df_2h: pd.DataFrame, df_exec: pd.DataFrame) -> Optional[Dict]:
        """
        Génère un signal basé sur multi-timeframe, RSI, order block, BoS, Fibonacci et momentum.
        """
        if len(df_exec) < 20 or len(df_4h) < 20 or len(df_2h) < 20 or len(df_1d) < 20:
            return None

        trend = self.detect_trend_24h(df_1d)
        if trend == "NEUTRAL":
            return None

        current_price = df_exec['close'].iloc[-1]
        levels = self.find_support_resistance(df_4h, df_2h)
        rsi_series = self.compute_rsi(df_exec['close'], period=14)
        macd_hist = self.compute_macd(df_exec['close']).iloc[-1]
        momentum = self.compute_momentum(df_exec['close'], period=8).iloc[-1]
        rsi_value = rsi_series.iloc[-1]
        trend_strength = float((df_1d['close'].ewm(span=20, adjust=False).mean().iloc[-1] - df_1d['close'].ewm(span=50, adjust=False).mean().iloc[-1]) / current_price)

        bos_dir, bos_price = self.detect_bos_retest(df_exec)
        fib_dir, fib_price = self.detect_fibonacci_retracement(df_exec)
        pattern_dir, pattern_name = self.detect_chart_patterns(df_exec)

        is_buy_ob = self.is_in_buy_order_block(df_4h, df_2h, df_exec)
        is_sell_ob = self.is_in_sell_order_block(df_4h, df_2h, df_exec)

        buy_score = 0
        sell_score = 0

        if trend == "BULLISH":
            buy_score += 1
            if rsi_value < 50:
                buy_score += 1
            if is_buy_ob:
                buy_score += 2
            if bos_dir == "BUY":
                buy_score += 1
            if fib_dir == "BUY":
                buy_score += 1
            if pattern_dir == "BUY":
                buy_score += 1
            if macd_hist > 0:
                buy_score += 1
            if momentum > 0:
                buy_score += 1
        elif trend == "BEARISH":
            sell_score += 1
            if rsi_value > 50:
                sell_score += 1
            if is_sell_ob:
                sell_score += 2
            if bos_dir == "SELL":
                sell_score += 1
            if fib_dir == "SELL":
                sell_score += 1
            if pattern_dir == "SELL":
                sell_score += 1
            if macd_hist < 0:
                sell_score += 1
            if momentum < 0:
                sell_score += 1

        direction = None
        strategy_used = None
        trigger_price = current_price

        if buy_score >= 4:
            direction = "BUY"
            strategy_used = "RSI+OB+MultiTF"
            trigger_price = bos_price or fib_price or current_price
        elif sell_score >= 4:
            direction = "SELL"
            strategy_used = "RSI+OB+MultiTF"
            trigger_price = bos_price or fib_price or current_price
        else:
            return None

        atr = self.calculate_atr(df_exec, period=14)
        if atr <= 0:
            atr = max(current_price * 0.0008, 0.001)

        if direction == "BUY":
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3.0)
        else:
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3.0)

        return {
            "symbol": "__DYNAMIC__",
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "strategy": strategy_used,
            "trend_24h": trend,
            "features": {
                "trend_slope": float((df_1d['close'].ewm(span=50, adjust=False).mean().iloc[-1] - df_1d['close'].ewm(span=50, adjust=False).mean().iloc[-3]) / 3),
                "trend_strength": trend_strength,
                "dist_support": float(min([abs(current_price - s) for s in levels["supports"]]) / current_price) if levels["supports"] else 1.0,
                "dist_resistance": float(min([abs(current_price - r) for r in levels["resistances"]]) / current_price) if levels["resistances"] else 1.0,
                "is_order_block": 1.0 if (is_buy_ob or is_sell_ob) else 0.0,
                "is_fib_zone": 1.0 if fib_dir else 0.0,
                "is_bos": 1.0 if bos_dir else 0.0,
                "pattern_detected": 1.0 if pattern_dir else 0.0,
                "rsi": float(rsi_value),
                "macd_hist": float(macd_hist),
                "momentum": float(momentum),
                "hour_of_day": int(df_exec.index[-1].hour) if isinstance(df_exec.index[-1], pd.Timestamp) else 12,
                "atr": float(atr)
            }
        }
