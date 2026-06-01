import os
import time
import random
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExecutionEngine")

BROKER_MARKETS = {
    "mt5": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "XAUUSD"],
    "simulation": ["EURUSD", "GBPUSD", "XAUUSD", "Volatility 75 Index", "Boom 500 Index"],
}

MT5_AVAILABLE = False
mt5 = None
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.warning("Bibliothèque MetaTrader5 non trouvée. Mode Simulation uniquement pour MT5.")
except Exception as e:
    logger.error(f"Erreur d'initialisation MT5: {e}")


class ExecutionEngine:
    def __init__(self, learning_engine, data_dir: str = "."):
        self.learning_engine = learning_engine
        self.data_dir = data_dir
        self.active_trades: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.current_broker = "simulation"
        self.broker_markets = BROKER_MARKETS.copy()
        self.active_markets = self.broker_markets[self.current_broker]
        self.selected_market = self.active_markets[0]
        self.simulation_mode = True
        
        # MT5 configuration
        self.mt5_account_id = None
        self.mt5_password = None
        self.mt5_server = None
        self.mt5_connected = False
        self.mt5_error_message = None

        # Forex & Deriv simulated prices (for demo mode)
        self.simulated_prices: Dict[str, float] = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2720,
            "USDJPY": 148.25,
            "AUDUSD": 0.6750,
            "NZDUSD": 0.6100,
            "XAUUSD": 2350.0,
            "Volatility 75 Index": 245000.0,
            "Volatility 100 Index": 1850.0,
            "Boom 500 Index": 4850.0,
            "Crash 1000 Index": 7250.0
        }
        self._load_closed_trades()
        self.simulated_history: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._initialize_simulated_history()

        self.selected_broker = "simulation"
        self.active_markets = BROKER_MARKETS[self.selected_broker]

        
        # Temps de dernière mise à jour pour génération de ticks en temps réel
        self.last_tick_time = time.time()
        self.tick_counter = 0

    def _load_closed_trades(self):
        closed_file = os.path.join(self.data_dir, "closed_trades.json")
        if os.path.exists(closed_file):
            try:
                import json
                with open(closed_file, 'r') as f:
                    self.closed_trades = json.load(f)
            except Exception:
                self.closed_trades = []

    def _save_closed_trades(self):
        import json
        closed_file = os.path.join(self.data_dir, "closed_trades.json")
        os.makedirs(self.data_dir, exist_ok=True)
        with open(closed_file, 'w') as f:
            json.dump(self.closed_trades, f, indent=4)

    def _initialize_simulated_history(self):
        """Génère un historique de prix initial simulé pour chaque symbole et timeframe."""
        timeframes = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
        now = datetime.now()
        
        for symbol, base_price in self.simulated_prices.items():
            self.simulated_history[symbol] = {}
            for tf in timeframes:
                # Nombre de bougies basé sur le timeframe (pour avoir ~1 mois de données)
                if tf == "1m":
                    n_candles = 1440  # ~1 jour
                    delta = timedelta(minutes=1)
                elif tf == "5m":
                    n_candles = 288   # ~1 jour
                    delta = timedelta(minutes=5)
                elif tf == "15m":
                    n_candles = 96    # ~1 jour
                    delta = timedelta(minutes=15)
                elif tf == "30m":
                    n_candles = 48    # ~1 jour
                    delta = timedelta(minutes=30)
                elif tf == "1h":
                    n_candles = 168   # ~1 semaine
                    delta = timedelta(hours=1)
                elif tf == "2h":
                    n_candles = 84    # ~1 semaine
                    delta = timedelta(hours=2)
                elif tf == "4h":
                    n_candles = 60    # ~10 jours
                    delta = timedelta(hours=4)
                else:  # 1d
                    n_candles = 60    # ~2 mois
                    delta = timedelta(days=1)
                    
                times = [now - (n_candles - i) * delta for i in range(n_candles)]
                
                trend = random.choice([-0.0001, 0.0001]) * base_price
                prices = [base_price]
                for _ in range(1, n_candles):
                    change = (random.normalvariate(0, 0.001) * base_price) + trend
                    prices.append(prices[-1] + change)
                    
                df = pd.DataFrame(index=times)
                df['close'] = prices
                df['open'] = df['close'].shift(1).fillna(base_price)
                df['high'] = df[['open', 'close']].max(axis=1) + (np.abs(np.random.normal(0, 0.0005, n_candles)) * base_price)
                df['low'] = df[['open', 'close']].min(axis=1) - (np.abs(np.random.normal(0, 0.0005, n_candles)) * base_price)
                df['volume'] = np.random.randint(100, 1000, n_candles)
                
                self.simulated_history[symbol][tf] = df

    def _find_mt5_terminal(self) -> Optional[str]:
        """Recherche le chemin du terminal MetaTrader 5 sur Windows."""
        candidates = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files\MetaTrader 5\terminal.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def setup_mt5(self, account_id: str, password: str, server: str) -> bool:
        """Configure la connexion MT5 (Exness Demo / MetaTrader 5)."""
        if not MT5_AVAILABLE:
            logger.error("MT5 API n'est pas disponible. Installez MetaTrader5 ou utilisez le mode Simulation.")
            return False

        try:
            self.mt5_error_message = None

            # Fermer toute instance MT5 précédente
            try:
                if mt5.initialize() is True:
                    mt5.shutdown()
            except Exception:
                pass

            init_success = mt5.initialize()
            terminal_path = None
            if not init_success:
                mt5.shutdown()
                terminal_path = self._find_mt5_terminal()
                if terminal_path:
                    logger.info(f"Essai d'initialisation MT5 avec le terminal '{terminal_path}'")
                    init_success = mt5.initialize(path=terminal_path)
                else:
                    logger.warning("Aucun terminal MT5 trouvé dans les chemins standards.")

            if not init_success:
                err = mt5.last_error()
                self.mt5_error_message = (
                    f"MT5 init failed. Vérifiez que MetaTrader 5 est installé et lancé. "
                    f"Chemin détecté: {terminal_path or 'aucun'}; erreur MT5: {err}"
                )
                logger.error(self.mt5_error_message)
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                return False

            login = int(account_id) if str(account_id).isdigit() else account_id
            authorized = mt5.login(login, password=password, server=server)
            if not authorized:
                err = mt5.last_error()
                self.mt5_error_message = f"Erreur de connexion MT5 (login): {err}"
                logger.error(self.mt5_error_message)
                mt5.shutdown()
                return False

            self.mt5_account_id = str(account_id)
            self.mt5_password = password
            self.mt5_server = server
            self.mt5_connected = True

            # Récupération dynamique des symboles de ce courtier
            try:
                all_symbols = mt5.symbols_get()
                if all_symbols:
                    # Prendre les symboles sélectionnés/visibles dans la liste d'observation du terminal
                    detected = [s.name for s in all_symbols if s.visible]
                    if not detected:
                        # Fallback : prendre les 30 premiers symboles du terminal
                        detected = [s.name for s in all_symbols[:30]]
                    
                    self.broker_markets["mt5"] = detected
                    logger.info(f"Symboles MT5 détectés : {detected}")
                else:
                    self.broker_markets["mt5"] = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "XAUUSD"]
            except Exception as se:
                logger.warning(f"Erreur lors de la récupération des symboles MT5 : {se}")
                self.broker_markets["mt5"] = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "XAUUSD"]

            self.current_broker = "mt5"
            self.active_markets = self.broker_markets["mt5"]
            self.selected_market = self.active_markets[0]
            self.simulation_mode = False
            logger.info(f"Connecté à MT5 {server} avec succès.")
            self._save_mt5_config()
            return True
        except Exception as e:
            logger.error(f"Erreur de connexion MT5: {e}")
            try:
                mt5.shutdown()
            except Exception:
                pass
            return False

    def _save_mt5_config(self):
        """Sauvegarde les paramètres MT5 dans mt5_config.json avec masquage base64 pour le mot de passe."""
        config_file = os.path.join(self.data_dir, "mt5_config.json")
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            import json
            import base64
            encoded_password = base64.b64encode(self.mt5_password.encode()).decode() if self.mt5_password else ""
            with open(config_file, "w") as f:
                json.dump({
                    "account_id": self.mt5_account_id,
                    "password_b64": encoded_password,
                    "server": self.mt5_server
                }, f, indent=4)
            logger.info("Configuration MT5 sauvegardée localement.")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la config MT5: {e}")

    def load_mt5_config(self) -> bool:
        """Charge la configuration MT5 et tente la connexion automatique."""
        config_file = os.path.join(self.data_dir, "mt5_config.json")
        if os.path.exists(config_file):
            try:
                import json
                import base64
                with open(config_file, "r") as f:
                    data = json.load(f)
                account_id = data.get("account_id")
                pwd_b64 = data.get("password_b64", "")
                password = base64.b64decode(pwd_b64.encode()).decode() if pwd_b64 else ""
                server = data.get("server")
                
                if account_id and password and server:
                    logger.info(f"Tentative de reconnexion automatique à MT5 (Compte: {account_id}, Serveur: {server})...")
                    return self.setup_mt5(account_id, password, server)
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la config MT5: {e}")
        return False


    def fetch_data(self, symbol: str, timeframe: Optional[str] = None):
        """Récupère les données OHLCV pour un timeframe spécifique ou plusieurs timeframes de base."""
        if timeframe is None:
            if not self.simulation_mode and self.mt5_connected:
                try:
                    d1 = self._fetch_mt5_candles(symbol, mt5.TIMEFRAME_D1, 100)
                    h4 = self._fetch_mt5_candles(symbol, mt5.TIMEFRAME_H4, 100)
                    h2 = self._fetch_mt5_candles(symbol, mt5.TIMEFRAME_H2, 100)
                    m1 = self._fetch_mt5_candles(symbol, mt5.TIMEFRAME_M1, 100)
                    return d1, h4, h2, m1
                except Exception as e:
                    logger.error(f"Erreur fetch MT5 {symbol}: {e}")
                    self.simulation_mode = True

            if symbol in self.simulated_history:
                hist = self.simulated_history[symbol]
                return hist["1d"], hist["4h"], hist["2h"], hist["1m"]
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        supported = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
        if timeframe not in supported:
            logger.warning(f"Timeframe non supporté: {timeframe}")
            timeframe = "1d"

        if not self.simulation_mode and self.mt5_connected:
            try:
                mt5_tf = self._timeframe_to_mt5(timeframe)
                return self._fetch_mt5_candles(symbol, mt5_tf, 150)
            except Exception as e:
                logger.error(f"Erreur fetch MT5 {symbol} {timeframe}: {e}")
                self.simulation_mode = True

        if symbol in self.simulated_history:
            return self.simulated_history[symbol].get(timeframe, pd.DataFrame())
        return pd.DataFrame()

    def _timeframe_to_mt5(self, timeframe: str) -> int:
        mapping = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "2h": mt5.TIMEFRAME_H2,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        return mapping.get(timeframe, mt5.TIMEFRAME_D1)

    def _normalize_mt5_symbol(self, symbol: str) -> str:
        """Tente de trouver le nom exact du symbole sur le courtier MT5 connecté."""
        if not self.mt5_connected or not MT5_AVAILABLE:
            return symbol

        # 1. Tester la correspondance exacte
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                return symbol
        except Exception:
            pass

        # 2. Recherche par correspondance floue (sans espaces, tirets, casse ou suffixes)
        try:
            all_symbols = mt5.symbols_get()
            if all_symbols:
                target_clean = symbol.lower().replace(" ", "").replace("_", "").replace("-", "")
                # Enlever les suffixes standards comme .m, .ecn, .demo
                for suffix in [".m", ".ecn", ".demo", ".raw"]:
                    if target_clean.endswith(suffix):
                        target_clean = target_clean[:-len(suffix)]

                for s in all_symbols:
                    s_clean = s.name.lower().replace(" ", "").replace("_", "").replace("-", "")
                    # Comparer la racine du symbole
                    if target_clean in s_clean or s_clean in target_clean:
                        logger.info(f"[MAP] Correspondance floue de symbole trouvée : '{symbol}' -> '{s.name}'")
                        return s.name
        except Exception as e:
            logger.warning(f"Erreur de mappage flou de symbole : {e}")

        return symbol

    def _fetch_mt5_candles(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        """Récupère les candles depuis MT5."""
        try:
            symbol = self._normalize_mt5_symbol(symbol)
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Symbole MT5 non sélectionné: {symbol}")

            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df.rename(columns={'tick_volume': 'volume'})
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.error(f"Erreur _fetch_mt5_candles {symbol}: {e}")
            return pd.DataFrame()

    def execute_trade(self, symbol: str, direction: str, lot: float, sl: float, tp: float, signal_features: Dict, strategy_name: str) -> Optional[str]:
        """Exécute un ordre d'achat ou de vente."""
        trade_id = f"T_{int(time.time())}_{random.randint(100, 999)}"
        entry_price = self.get_current_price(symbol)

        trade = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "lot": lot,
            "entry_price": entry_price,
            "stop_loss": sl,
            "take_profit": tp,
            "strategy": strategy_name,
            "open_time": datetime.now().isoformat(),
            "status": "OPEN",
            "features": signal_features
        }

        if not self.simulation_mode and self.mt5_connected:
            try:
                symbol = self._normalize_mt5_symbol(symbol)
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    logger.error(f"Impossible de récupérer le tick MT5 pour {symbol}.")
                    return None

                price = float(tick.ask if direction == "BUY" else tick.bid)
                order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(lot),
                    "type": order_type,
                    "price": price,
                    "sl": float(sl),
                    "tp": float(tp),
                    "deviation": 20,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                    "type_time": mt5.ORDER_TIME_GTC,
                }
                result = mt5.order_send(request)
                if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Erreur exécution MT5: {getattr(result, 'comment', result)}")
                    return None
                trade["mt5_ticket"] = getattr(result, 'order', None)
                logger.info(f"Trade MT5 exécuté: {direction} {symbol} x{lot} @ {entry_price}")
            except Exception as e:
                logger.error(f"Erreur execute_trade MT5: {e}")
                return None

        self.active_trades.append(trade)
        logger.info(f"Nouveau trade ouvert: {direction} {symbol} Lot:{lot} Entrée:{entry_price:.5f} SL:{sl:.5f} TP:{tp:.5f}")
        return trade_id

    def get_current_price(self, symbol: str) -> float:
        if not self.simulation_mode and self.mt5_connected:
            try:
                symbol = self._normalize_mt5_symbol(symbol)
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    return float(tick.bid if tick.bid else tick.ask)
            except Exception as e:
                logger.error(f"Erreur get_current_price MT5: {e}")

        return self.simulated_prices.get(symbol, 1.0)

    def _generate_live_ticks(self):
        """Génère de nouveaux ticks en temps réel pour créer de nouvelles bougies 1m chaque seconde."""
        if not self.simulation_mode:
            return
        
        now = time.time()
        seconds_elapsed = now - self.last_tick_time
        
        # Générer 1 nouveau tick/bougie chaque seconde
        if seconds_elapsed < 1.0:
            return
        
        self.last_tick_time = now
        self.tick_counter += 1
        
        # Mettre à jour chaque symbole avec une nouvelle bougie 1m
        current_time = datetime.now()
        
        for symbol in list(self.simulated_history.keys()):
            hist = self.simulated_history[symbol]
            if '1m' not in hist or len(hist['1m']) == 0:
                continue
            
            last_row = hist['1m'].iloc[-1]
            last_close = float(last_row['close'])
            last_price = self.simulated_prices.get(symbol, last_close)
            
            # Générer un mouvement petit et réaliste
            volatility = 0.0008 * last_price  # 0.08% de volatilité
            trend = (random.random() - 0.5) * 0.0002 * last_price
            price_change = np.random.normal(trend, volatility)
            
            new_close = last_price + price_change
            new_close = max(new_close, last_price * 0.99)  # Pas de crash
            new_close = min(new_close, last_price * 1.01)  # Limiter les pics
            
            # Créer la nouvelle bougie 1m
            new_open = last_close
            new_high = max(new_open, new_close) + abs(np.random.normal(0, volatility * 0.3))
            new_low = min(new_open, new_close) - abs(np.random.normal(0, volatility * 0.3))
            new_volume = np.random.randint(50, 500)
            
            # Créer une nouvelle ligne pour 1m
            new_row = pd.DataFrame({
                'open': [new_open],
                'high': [new_high],
                'low': [new_low],
                'close': [new_close],
                'volume': [new_volume]
            }, index=[current_time])
            
            # Ajouter à l'historique et limiter à 1500 bougies (~ 1 jour)
            hist['1m'] = pd.concat([hist['1m'], new_row])
            if len(hist['1m']) > 1500:
                hist['1m'] = hist['1m'].iloc[-1500:]
            
            # Mettre à jour les prix actuels
            self.simulated_prices[symbol] = new_close
            
            # Mettre à jour les timeframes plus hauts chaque minute/5min/etc
            self._update_higher_timeframes(symbol, hist, current_time)
    
    def _update_higher_timeframes(self, symbol: str, hist: Dict, current_time: datetime):
        """Met à jour les timeframes 5m, 15m, 30m, 1h, 2h, 4h, 1d à partir de 1m."""
        if '1m' not in hist or len(hist['1m']) < 2:
            return
        
        # Les bougies 5m, 15m, etc. se créent automatiquement tous les 5, 15, 30 min...
        timeframe_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '1d': 1440}
        
        for tf_name, interval in timeframe_mins.items():
            if tf_name not in hist or len(hist[tf_name]) == 0:
                continue
            
            # Grouper les bougies 1m par timeframe
            one_min_data = hist['1m'].copy()
            one_min_data['time_bucket'] = one_min_data.index.floor(f'{interval}min')
            
            grouped = one_min_data.groupby('time_bucket')
            agg_data = grouped.agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(agg_data) > 0:
                # Remplacer l'historique du timeframe par les données agrégées
                hist[tf_name] = agg_data
                if len(hist[tf_name]) > 500:
                    hist[tf_name] = hist[tf_name].iloc[-500:]

    def tick_simulation(self):
        """Mise à jour des prix simulés avec génération de nouveaux ticks en temps réel."""
        # Générer de nouveaux ticks/bougies
        self._generate_live_ticks()
        
        # Mettre à jour les prix actuels depuis les dernières bougies
        if self.simulation_mode:
            for symbol in list(self.simulated_history.keys()):
                hist = self.simulated_history[symbol]
                if '1m' in hist and len(hist['1m']) > 0:
                    current = hist['1m'].iloc[-1]
                    self.simulated_prices[symbol] = float(current['close'])

    def set_market(self, market: str) -> bool:
        if market in self.broker_markets.get(self.current_broker, []):
            self.selected_market = market
            return True
        if market in self.simulated_prices:
            self.selected_market = market
            return True
        return False

    def get_available_markets(self, broker: str):
        return self.broker_markets.get(broker.lower(), list(self.simulated_prices.keys()))

    def disconnect_mt5(self):
        if self.mt5_connected:
            try:
                mt5.shutdown()
            except Exception:
                pass
            self.mt5_connected = False
        self.simulation_mode = True
        self.current_broker = "simulation"
        self.active_markets = self.broker_markets[self.current_broker]
        self.selected_market = self.active_markets[0]
        logger.info("Déconnecté de MT5. Mode simulation activé.")

    def set_broker(self, broker: str) -> bool:
        broker_key = broker.lower()
        if broker_key not in self.broker_markets:
            return False
        self.current_broker = broker_key
        self.active_markets = self.broker_markets[broker_key]
        self.selected_market = self.active_markets[0]
        self.simulation_mode = broker_key != "mt5" or not self.mt5_connected
        return True

