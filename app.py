import os
import time
import asyncio
import threading
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

import logging
logger = logging.getLogger("app")
from learning_engine import LearningEngine
from risk_manager import RiskManager
from execution_engine import ExecutionEngine
from strategy import StrategyAnalyzer

app = FastAPI(title="Trading AI Dashboard")

BROKER_KEY_MAP = {
    "Simulation": "simulation",
    "Exness MT5": "mt5",
    "MT5": "mt5"
}

DISPLAY_BROKER_NAME = {
    "simulation": "Simulation",
    "mt5": "Exness MT5"
}

# Activation du CORS pour le développement
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation des instances
DATA_DIR = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "scratch", "trading_ai_data")
learning_engine = LearningEngine(data_dir=DATA_DIR)
risk_manager = RiskManager(target_gain=0.50, max_loss=0.75)
execution_engine = ExecutionEngine(learning_engine=learning_engine, data_dir=DATA_DIR)
analyzer = StrategyAnalyzer()

# Statistiques globales
stats = {
    "blocked_trades_count": 0,
    "total_simulated_balance": 1000.0,
    "running": True,
    "bot_running": True,
    "selected_broker": execution_engine.current_broker,
    "selected_market": execution_engine.selected_market,
    "mt5_balance": None
}

# Modèles Pydantic
class MT5Config(BaseModel):
    account_id: str
    password: str
    server: str

class TargetConfig(BaseModel):
    target_gain: float
    max_loss: float
    simulation_mode: bool

class BrokerConfig(BaseModel):
    broker: str
    market: Optional[str] = None

class StartConfig(BaseModel):
    start: bool

class LicenseActivation(BaseModel):
    license_key: str

# Gestion de la licence Premium
LICENSE_FILE = os.path.join(DATA_DIR, "license_config.json")

def load_license_key() -> Optional[str]:
    if os.path.exists(LICENSE_FILE):
        try:
            with open(LICENSE_FILE, "r") as f:
                data = json.load(f)
                return data.get("license_key")
        except Exception:
            return None
    return None

def save_license_key(key: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LICENSE_FILE, "w") as f:
        json.dump({"license_key": key}, f)

def is_license_valid(key: Optional[str]) -> bool:
    if not key:
        return False
    return key == "NEURAL-TRADER-PRO-2026" or key.startswith("NT-PRO-")



# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE DU BOT EN ARRIÈRE-PLAN
# ─────────────────────────────────────────────────────────────────────────────
def bot_background_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tick_counter = 0
    COOLDOWN_TICKS = 30  # ~30 secondes de pause entre trades sur le même symbole
    symbol_cooldown: Dict[str, int] = {}

    while stats["running"]:
        try:
            if not stats["bot_running"]:
                time.sleep(1.0)
                continue

            # 1. Tick de simulation (variations de prix + vérification TP/SL)
            execution_engine.tick_simulation()

            tick_counter += 1
            if tick_counter >= 5:
                tick_counter = 0

                symbols = list(execution_engine.active_markets)
                for symbol in symbols:
                    if symbol not in symbol_cooldown:
                        symbol_cooldown[symbol] = 0

                # Diminuer les cooldowns
                for s in list(symbol_cooldown.keys()):
                    if symbol_cooldown.get(s, 0) > 0:
                        symbol_cooldown[s] -= 1

                # Limite globale du nombre de trades ouverts
                max_active = 2
                active_symbols = [t["symbol"] for t in execution_engine.active_trades]
                active_count = len(execution_engine.active_trades)
                print(f"[BOT] Vérification des trades : actifs={active_count}, max={max_active}, marchés={symbols}")

                for symbol in symbols:
                    # Ne pas retradre si déjà en position sur ce symbole
                    if symbol in active_symbols:
                        continue
                    # Ne pas retradre pendant le cooldown
                    if symbol_cooldown[symbol] > 0:
                        continue
                    # Limite globale du nombre de trades ouverts
                    if active_count >= max_active:
                        break

                    try:
                        # Récupérer les données OHLCV
                        d1, h4, h2, exec_tf = execution_engine.fetch_data(symbol)

                        # Analyse technique
                        signal = analyzer.generate_signal(d1, h4, h2, exec_tf)

                        # Si le signal provient de la stratégie, injecter le bon symbole
                        if signal:
                            signal["symbol"] = symbol
                            print(f"[BOT] Signal technique détecté : {signal['direction']} {symbol} [{signal.get('strategy', 'unknown')}]")

                        # En mode simulation : générer un signal démo si aucun signal technique
                        elif execution_engine.simulation_mode and active_count < max_active:
                            import random
                            current_price = execution_engine.get_current_price(symbol)
                            atr = current_price * 0.002
                            direction = random.choice(["BUY", "SELL"])

                            signal = {
                                "symbol": symbol,
                                "direction": direction,
                                "entry_price": current_price,
                                "stop_loss": (
                                    current_price - atr * 1.5 if direction == "BUY"
                                    else current_price + atr * 1.5
                                ),
                                "take_profit": (
                                    current_price + atr * 1.5 if direction == "BUY"
                                    else current_price - atr * 1.5
                                ),
                                "strategy": random.choice(
                                    ["BoS Retest", "Fibonacci Retracement", "Pattern Double Bottom"]
                                ),
                                "trend_24h": random.choice(["BULLISH", "BEARISH"]),
                                "features": {
                                    "trend_slope": random.uniform(-0.001, 0.001),
                                    "dist_support": random.uniform(0.0, 0.02),
                                    "dist_resistance": random.uniform(0.0, 0.02),
                                    "is_fib_zone": float(random.choice([0, 1])),
                                    "is_bos": float(random.choice([0, 1])),
                                    "pattern_detected": float(random.choice([0, 1])),
                                    "hour_of_day": datetime.now().hour,
                                    "atr": random.uniform(0.001, 0.005)
                                }
                            }
                            print(f"[BOT] Signal simulation généré : {direction} {symbol} [{signal['strategy']}]")

                        if signal:
                            # Validation IA
                            is_viable, probability = learning_engine.predict_trade_viability(
                                signal["features"]
                            )

                            if is_viable:
                                lot, sl, tp = risk_manager.calculate_position(
                                    symbol=signal["symbol"],
                                    direction=signal["direction"],
                                    entry=signal["entry_price"],
                                    sl=signal["stop_loss"]
                                )
                                trade_id = execution_engine.execute_trade(
                                    symbol=signal["symbol"],
                                    direction=signal["direction"],
                                    lot=lot,
                                    sl=sl,
                                    tp=tp,
                                    signal_features=signal["features"],
                                    strategy_name=signal["strategy"]
                                )
                                if trade_id:
                                    active_count += 1
                                    # Activer le cooldown sur ce symbole
                                    symbol_cooldown[symbol] = COOLDOWN_TICKS
                                    print(f"[BOT] Trade exécuté : {signal['direction']} {symbol} Lot={lot} SL={sl:.5f} TP={tp:.5f}")
                            else:
                                stats["blocked_trades_count"] += 1
                                print(
                                    f"[IA BLOQUÉ] {signal['direction']} {symbol} "
                                    f"rejeté (probabilité: {probability:.1%})"
                                )
                    except KeyError as ke:
                        print(f"[WARN] Symbole introuvable: {ke} — ignoré")
                    except Exception as e:
                        print(f"[ERREUR] Analyse {symbol}: {e}")

            time.sleep(1.0)

        except Exception as e:
            print(f"[ERREUR CRITIQUE] Boucle bot: {e}")
            time.sleep(3.0)


# Démarrage au lancement de l'application
@app.on_event("startup")
def startup_event():
    # Reconnexion automatique si licence valide
    key = load_license_key()
    if is_license_valid(key):
        print("[INFO] Licence Premium valide detectee. Tentative de reconnexion MT5...")
        success = execution_engine.load_mt5_config()
        if success:
            stats["selected_broker"] = execution_engine.current_broker
            stats["selected_market"] = execution_engine.selected_market
            print(f"[OK] Reconnecte a MT5: {execution_engine.mt5_server}")

    thread = threading.Thread(target=bot_background_loop, daemon=True)
    thread.start()
    print("[OK] Neural Trader IA demarre")
    print(f"[OK] Bot automatique actif : {stats['bot_running']}")

@app.on_event("shutdown")
def shutdown_event():
    stats["running"] = False


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS API REST
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    closed_trades = execution_engine.closed_trades
    total_profit = sum([t.get("profit", 0.0) for t in closed_trades])
    wins = sum([1 for t in closed_trades if t.get("outcome", 0) == 1])
    losses = sum([1 for t in closed_trades if t.get("outcome", 0) == 0])
    total = wins + losses
    win_rate = (wins / total) if total > 0 else 0.0
    
    # Get balance from MT5 if connected, otherwise use simulated balance
    if not execution_engine.simulation_mode and execution_engine.mt5_connected:
        try:
            from execution_engine import mt5
            if mt5 and hasattr(mt5, 'account_info'):
                account = mt5.account_info()
                if account:
                    current_balance = account.balance + total_profit
                    stats["mt5_balance"] = account.balance
                else:
                    current_balance = stats["total_simulated_balance"] + total_profit
            else:
                current_balance = stats["total_simulated_balance"] + total_profit
        except Exception as e:
            logger.warning(f"Failed to fetch MT5 balance: {e}")
            current_balance = stats["total_simulated_balance"] + total_profit
    else:
        current_balance = stats["total_simulated_balance"] + total_profit

    broker_display = DISPLAY_BROKER_NAME.get(execution_engine.current_broker, execution_engine.current_broker)
    mt5_connected = getattr(execution_engine, 'mt5_connected', False)

    return {
        "status": "RUNNING",
        "simulation_mode": execution_engine.simulation_mode,
        "active_trades_count": len(execution_engine.active_trades),
        "closed_trades_count": len(closed_trades),
        "blocked_trades_count": stats["blocked_trades_count"],
        "total_profit": round(total_profit, 2),
        "win_rate": round(win_rate * 100, 1),
        "balance": round(current_balance, 2),
        "wins": wins,
        "losses": losses,
        "target_gain": risk_manager.target_gain,
        "max_loss": risk_manager.max_loss,
        "bot_running": stats["bot_running"],
        "selected_broker": broker_display,
        "selected_market": execution_engine.selected_market,
        "available_markets": execution_engine.get_available_markets(execution_engine.current_broker),
        "mt5_connected": mt5_connected
    }

@app.get("/api/trades/active")
def get_active_trades():
    return execution_engine.active_trades

@app.get("/api/trades/closed")
def get_closed_trades():
    return list(reversed(execution_engine.closed_trades))[-20:]

@app.get("/api/insights")
def get_insights():
    return {"insights": learning_engine.generate_ai_insights()}

@app.post("/api/config/target")
def update_target(config: TargetConfig):
    risk_manager.target_gain = config.target_gain
    risk_manager.max_loss = config.max_loss
    execution_engine.simulation_mode = config.simulation_mode
    if config.simulation_mode:
        execution_engine.disconnect_mt5()
    return {"message": "Configuration mise à jour"}

@app.get("/api/brokers")
def get_brokers():
    return {
        "brokers": ["Simulation", "Exness MT5"],
        "markets": {
            "Simulation": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "XAUUSD"],
            "Exness MT5": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "XAUUSD"]
        }
    }

@app.post("/api/config/broker")
def update_broker(config: BrokerConfig):
    broker = BROKER_KEY_MAP.get(config.broker, (config.broker or "").lower())
    market = config.market
    
    # Restreindre le passage en réel si pas de licence
    if broker == "mt5":
        key = load_license_key()
        if not is_license_valid(key):
            execution_engine.set_broker("simulation")
            stats["selected_broker"] = "simulation"
            return {
                "message": "Licence Premium requise pour passer en mode réel MT5.",
                "selected_broker": "Simulation",
                "selected_market": execution_engine.selected_market,
                "simulation_mode": True,
                "available_markets": execution_engine.get_available_markets("simulation"),
                "error": "LICENSE_REQUIRED"
            }

    execution_engine.set_broker(broker)
    stats["selected_broker"] = execution_engine.current_broker
    if market:
        execution_engine.set_market(market)
    stats["selected_market"] = execution_engine.selected_market

    display_broker = DISPLAY_BROKER_NAME.get(stats["selected_broker"], stats["selected_broker"])

    return {
        "message": "Broker mis à jour",
        "selected_broker": display_broker,
        "selected_market": stats["selected_market"],
        "simulation_mode": execution_engine.simulation_mode,
        "available_markets": execution_engine.get_available_markets(execution_engine.current_broker)
    }

@app.post("/api/config/mt5")
def configure_mt5(config: MT5Config):
    # Vérifier la licence
    key = load_license_key()
    if not is_license_valid(key):
        return {
            "status": "FAILED",
            "message": "Clé de licence Premium requise pour se connecter à un compte réel MetaTrader 5.",
            "details": "Veuillez activer une licence Premium valide."
        }

    success = execution_engine.setup_mt5(
        account_id=config.account_id,
        password=config.password,
        server=config.server
    )
    if success:
        stats["selected_broker"] = execution_engine.current_broker
        stats["selected_market"] = execution_engine.selected_market
        return {"status": "SUCCESS", "message": "MT5 configuré avec succès."}

    error_msg = getattr(execution_engine, 'mt5_error_message', None)
    if not error_msg:
        error_msg = "Échec de configuration MT5. Causes possibles: (1) MetaTrader 5 n'est pas installé/lancé, (2) Identifiant/mot de passe/serveur incorrect."

    return {
        "status": "FAILED",
        "message": error_msg,
        "details": error_msg
    }

@app.get("/api/license/status")
def get_license_status():
    key = load_license_key()
    valid = is_license_valid(key)
    return {
        "license_key": key,
        "is_valid": valid,
        "status": "Premium Active" if valid else "Version Gratuite (Simulation)"
    }

@app.post("/api/license/activate")
def activate_license(data: LicenseActivation):
    if is_license_valid(data.license_key):
        save_license_key(data.license_key)
        return {"status": "SUCCESS", "message": "Licence Premium activée !"}
    return {"status": "FAILED", "message": "Clé de licence invalide."}


@app.post("/api/config/start")
def start_bot(config: StartConfig):
    stats["bot_running"] = config.start
    return {"bot_running": stats["bot_running"]}

@app.get("/api/chart/{symbol}")
def get_chart_data(symbol: str, timeframe: str = "1d"):
    """Retourne les bougies OHLCV récentes pour le graphique TradingView pour le timeframe demandé."""
    try:
        df_exec = execution_engine.fetch_data(symbol, timeframe)
    except KeyError:
        return []

    chart_candles = []
    decimals = 2 if symbol == "XAUUSD" else 5
    for index, row in df_exec.iterrows():
        try:
            ts = int(index.timestamp())
        except Exception:
            ts = int(time.time())
        chart_candles.append({
            "time": ts,
            "open": round(float(row["open"]), decimals),
            "high": round(float(row["high"]), decimals),
            "low": round(float(row["low"]), decimals),
            "close": round(float(row["close"]), decimals)
        })

    # Dédupliquer les timestamps (TradingView exige des timestamps uniques)
    seen = set()
    unique_candles = []
    for c in chart_candles:
        if c["time"] not in seen:
            seen.add(c["time"])
            unique_candles.append(c)

    return sorted(unique_candles, key=lambda x: x["time"])

@app.get("/api/chart/{symbol}/latest")
def get_latest_chart_data(symbol: str, timeframe: str = "1d"):
    """Retourne la dernière bougie OHLCV pour le timeframe demandé."""
    try:
        df_exec = execution_engine.fetch_data(symbol, timeframe)
    except KeyError:
        return {}

    if df_exec.empty:
        return {}

    last_row = df_exec.iloc[-1]
    try:
        ts = int(last_row.name.timestamp())
    except Exception:
        ts = int(time.time())

    decimals = 2 if symbol == "XAUUSD" else 5
    return {
        "time": ts,
        "open": round(float(last_row["open"]), decimals),
        "high": round(float(last_row["high"]), decimals),
        "low": round(float(last_row["low"]), decimals),
        "close": round(float(last_row["close"]), decimals)
    }


def _format_sse(event: str, data: str) -> str:
    payload = f"event: {event}\n"
    for line in data.splitlines():
        payload += f"data: {line}\n"
    payload += "\n"
    return payload

@app.get("/sse/updates")
def sse_updates(symbol: str = "EURUSD", timeframe: str = "1d"):
    async def event_generator():
        last_active_ids = set()
        while True:
            try:
                status_data = get_status()
                yield _format_sse("status_update", json.dumps(status_data))

                active_trades = execution_engine.active_trades
                closed_trades = list(reversed(execution_engine.closed_trades))[:20]
                current_ids = {t["trade_id"] for t in active_trades}
                new_ids = current_ids - last_active_ids
                if new_ids:
                    for trade in active_trades:
                        if trade["trade_id"] in new_ids:
                            yield _format_sse("trade_opened", json.dumps(trade))
                last_active_ids = current_ids

                yield _format_sse("trades_update", json.dumps({
                    "active_trades": active_trades,
                    "closed_trades": closed_trades
                }))

                try:
                    df_exec = execution_engine.fetch_data(symbol, timeframe)
                    if not df_exec.empty:
                        last_row = df_exec.iloc[-1]
                        ts = int(last_row.name.timestamp()) if hasattr(last_row.name, 'timestamp') else int(time.time())
                        decimals = 2 if symbol == "XAUUSD" else 5
                        yield _format_sse("latest_candle", json.dumps({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "data": {
                                "time": ts,
                                "open": round(float(last_row["open"]), decimals),
                                "high": round(float(last_row["high"]), decimals),
                                "low": round(float(last_row["low"]), decimals),
                                "close": round(float(last_row["close"]), decimals)
                            }
                        }))
                except Exception:
                    pass
            except GeneratorExit:
                break
            except Exception:
                pass
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.websocket("/ws/chart")
async def websocket_chart_endpoint(websocket: WebSocket):
    await websocket.accept()
    symbol = "EURUSD"
    timeframe = "1d"

    async def receive_subscriptions():
        nonlocal symbol, timeframe
        try:
            while True:
                msg = await websocket.receive_json()
                if isinstance(msg, dict) and msg.get("action") == "subscribe":
                    symbol = msg.get("symbol", symbol)
                    timeframe = msg.get("timeframe", timeframe)
        except WebSocketDisconnect:
            raise
        except Exception:
            pass

    async def send_updates():
        last_active_ids = set()
        while True:
            try:
                status_data = get_status()
                await websocket.send_json({"type": "status_update", "data": status_data})

                active_trades = execution_engine.active_trades
                closed_trades = list(reversed(execution_engine.closed_trades))[:20]
                current_ids = {t["trade_id"] for t in active_trades}
                new_ids = current_ids - last_active_ids
                if new_ids:
                    for trade in active_trades:
                        if trade["trade_id"] in new_ids:
                            await websocket.send_json({"type": "trade_opened", "data": trade})
                last_active_ids = current_ids

                await websocket.send_json({
                    "type": "trades_update",
                    "data": {
                        "active_trades": active_trades,
                        "closed_trades": closed_trades
                    }
                })

                try:
                    df_exec = execution_engine.fetch_data(symbol, timeframe)
                    if not df_exec.empty:
                        last_row = df_exec.iloc[-1]
                        ts = int(last_row.name.timestamp()) if hasattr(last_row.name, 'timestamp') else int(time.time())
                        decimals = 2 if symbol == "XAUUSD" else 5
                        await websocket.send_json({
                            "type": "latest_candle",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "data": {
                                "time": ts,
                                "open": round(float(last_row["open"]), decimals),
                                "high": round(float(last_row["high"]), decimals),
                                "low": round(float(last_row["low"]), decimals),
                                "close": round(float(last_row["close"]), decimals)
                            }
                        })
                except Exception:
                    pass
            except WebSocketDisconnect:
                break
            except Exception:
                pass
            await asyncio.sleep(1.0)

    sender = asyncio.create_task(send_updates())
    receiver = asyncio.create_task(receive_subscriptions())
    try:
        await asyncio.gather(sender, receiver)
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        receiver.cancel()

@app.get("/api/symbols")
def get_symbols():
    """Retourne la liste des symboles disponibles avec leur prix actuel."""
    symbols = execution_engine.get_available_markets(execution_engine.current_broker)
    result = []
    for s in symbols:
        price = execution_engine.get_current_price(s)
        result.append({"symbol": s, "price": round(price, 5)})
    return result

# Fichiers statiques frontend
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
app.mount(
    "/",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True),
    name="static"
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
