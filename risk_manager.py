from typing import Dict, Tuple

class RiskManager:
    def __init__(self, target_gain: float = 0.50, max_loss: float = 1.00):
        self.target_gain = target_gain
        self.max_loss = max_loss
        
        # Spécifications des contrats Forex uniquement
        # (1 lot standard = 100 000 unités pour les paires majeurs)
        self.symbol_specs = {
            "EURUSD": {"contract_size": 100000.0, "min_lot": 0.01, "max_lot": 10.0, "is_forex": True},
            "GBPUSD": {"contract_size": 100000.0, "min_lot": 0.01, "max_lot": 10.0, "is_forex": True},
            "USDJPY": {"contract_size": 100000.0, "min_lot": 0.01, "max_lot": 10.0, "is_forex": True},
            "AUDUSD": {"contract_size": 100000.0, "min_lot": 0.01, "max_lot": 10.0, "is_forex": True},
            "NZDUSD": {"contract_size": 100000.0, "min_lot": 0.01, "max_lot": 10.0, "is_forex": True},
            "XAUUSD": {"contract_size": 100.0, "min_lot": 0.01, "max_lot": 5.0, "is_forex": True},
            "Volatility75Index": {"contract_size": 1.0, "min_lot": 0.005, "max_lot": 10.0, "is_forex": False},
            "Volatility100Index": {"contract_size": 1.0, "min_lot": 0.10, "max_lot": 50.0, "is_forex": False},
            "Boom500Index": {"contract_size": 1.0, "min_lot": 0.10, "max_lot": 100.0, "is_forex": False},
            "Crash1000Index": {"contract_size": 1.0, "min_lot": 0.10, "max_lot": 100.0, "is_forex": False},
        }

    def calculate_position(self, symbol: str, direction: str, entry: float, sl: float) -> Tuple[float, float, float]:
        """
        Calcule la taille du lot (Lot Size) pour cibler exactement le gain défini (ex: 0.5$),
        et ajuste le Take Profit (TP) et le Stop Loss (SL) pour respecter le budget de risque.
        
        Retourne (lot_size, adjusted_sl, adjusted_tp).
        """
        normalized_symbol = symbol.replace('_', '')
        spec = self.symbol_specs.get(normalized_symbol, self.symbol_specs.get(symbol, {"contract_size": 1.0, "min_lot": 0.01, "max_lot": 1.0, "is_forex": False}))
        contract_size = spec["contract_size"]
        min_lot = spec["min_lot"]
        max_lot = spec["max_lot"]
        
        # 1. Calculer la distance SL suggérée en valeur absolue
        sl_distance = abs(entry - sl)
        if sl_distance == 0:
            sl_distance = entry * 0.001 # Sécurité
            
        # 2. Définir la distance TP. Par défaut, on maintient un ratio Risk/Reward de 1:1.
        # Donc, distance TP = distance SL.
        # Cela signifie que si le TP est touché on gagne target_gain (0.5$),
        # et si le SL est touché on perd max_loss (par exemple 0.5$ ou 0.75$).
        tp_distance = sl_distance
        
        # Pour le Forex, ajustement de la valeur du pip
        if spec.get("is_forex", False):
            # Distance en pips/points (ex: EURUSD de 1.0850 à 1.0840 = 0.0010 = 10 pips)
            # Formule de gain Forex : Lot * ContractSize * distance_prix = Gain_USD
            # Donc Lot = Gain_USD / (ContractSize * distance_prix)
            raw_lot = self.target_gain / (contract_size * tp_distance)
        else:
            raw_lot = self.target_gain / (contract_size * tp_distance)
            
        # 3. Borner le lot calculé entre le min_lot et le max_lot
        lot_size = round(max(min(raw_lot, max_lot), min_lot), 3)
        
        # 4. Ajuster le TP et le SL exacts en fonction du lot final appliqué pour respecter les montants cibles
        # Gain = LotSize * ContractSize * TP_distance = target_gain
        # => TP_distance = target_gain / (LotSize * ContractSize)
        actual_tp_distance = self.target_gain / (lot_size * contract_size)
        
        # Perte = LotSize * ContractSize * SL_distance.
        # On souhaite que la perte ne dépasse pas max_loss (ex: 1.00$).
        # Donc SL_distance_max = max_loss / (LotSize * contract_size)
        # On prend le minimum entre le SL technique initial et le SL maximum autorisé financièrement
        max_technical_sl_distance = self.max_loss / (lot_size * contract_size)
        actual_sl_distance = min(sl_distance, max_technical_sl_distance)
        
        if direction == "BUY":
            adjusted_tp = entry + actual_tp_distance
            adjusted_sl = entry - actual_sl_distance
        else:
            adjusted_tp = entry - actual_tp_distance
            adjusted_sl = entry + actual_sl_distance
            
        return lot_size, adjusted_sl, adjusted_tp
