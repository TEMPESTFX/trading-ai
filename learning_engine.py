import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from typing import Dict, List, Tuple, Optional

class LearningEngine:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.history_file = os.path.join(data_dir, "trade_history.json")
        self.model_file = os.path.join(data_dir, "ai_model.pkl")
        self.trades = self._load_history()
        self.model = self._load_model()
        self.feature_names = [
            "trend_slope", "trend_strength", "dist_support", "dist_resistance",
            "is_order_block", "is_fib_zone", "is_bos", "pattern_detected",
            "rsi", "macd_hist", "momentum", "hour_of_day", "atr"
        ]

    def _load_history(self) -> List[Dict]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.trades, f, indent=4)

    def _load_model(self) -> Optional[RandomForestClassifier]:
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_model(self):
        if self.model:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)

    def register_trade_result(self, trade_id: str, features: Dict, outcome: int):
        """
        Enregistre le résultat d'un trade et réentraîne le modèle.
        outcome: 1 pour Gain (TP), 0 pour Perte (SL/annulé).
        """
        # Nettoyer les features pour ne garder que les numériques attendues
        cleaned_features = {k: float(features.get(k, 0.0)) for k in self.feature_names}
        
        self.trades.append({
            "trade_id": trade_id,
            "features": cleaned_features,
            "outcome": int(outcome)
        })
        self._save_history()
        self.retrain()

    def retrain(self) -> bool:
        """Réentraîne le modèle Random Forest sur l'historique complet des trades."""
        if len(self.trades) < 10:
            # Pas assez de données pour entraîner un modèle robuste
            return False
            
        df = pd.DataFrame([
            {**t['features'], 'outcome': t['outcome']}
            for t in self.trades
        ])
        
        # Vérifier qu'on a les deux classes (gagnant et perdant)
        if df['outcome'].nunique() < 2:
            return False
            
        X = df[self.feature_names]
        y = df['outcome']
        
        # Entraînement d'une forêt aléatoire
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)
        self.model = model
        self._save_model()
        return True

    def predict_trade_viability(self, features: Dict) -> Tuple[bool, float]:
        """
        Évalue si l'IA autorise le trade.
        Retourne (Autorisé: bool, Probabilité de succès: float).
        """
        if self.model is None or len(self.trades) < 10:
            # Avant 10 trades ou si le modèle n'est pas prêt, on laisse passer tous les signaux
            # pour accumuler des données d'apprentissage.
            return True, 1.0
            
        # Aligner les features
        x_input = [float(features.get(k, 0.0)) for k in self.feature_names]
        x_input = np.array(x_input).reshape(1, -1)
        
        # Obtenir la probabilité d'un gain (classe 1)
        prob = float(self.model.predict_proba(x_input)[0][1])
        
        # Filtrage par l'IA : n'autorise que les trades ayant au moins 52% de chances estimées de réussite
        authorized = prob >= 0.52
        return authorized, prob

    def generate_ai_insights(self) -> List[str]:
        """
        Analyse l'historique et le modèle pour générer des phrases claires expliquant
        ce que l'IA a appris des erreurs passées.
        """
        insights = []
        
        if len(self.trades) < 5:
            return ["L'IA est en phase d'observation initiale. Effectuez des trades pour commencer l'apprentissage."]
            
        df = pd.DataFrame([
            {**t['features'], 'outcome': t['outcome']}
            for t in self.trades
        ])
        
        total_trades = len(df)
        wins = df[df['outcome'] == 1]
        losses = df[df['outcome'] == 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        insights.append(f"Statistiques globales : {total_trades} trades enregistrés, Taux de réussite de {win_rate:.1%}.")
        
        # Analyser les pertes par heure
        if len(losses) > 0:
            bad_hours = losses['hour_of_day'].value_counts()
            if len(bad_hours) > 0 and bad_hours.iloc[0] >= 2:
                worst_hour = bad_hours.index[0]
                pct_loss = (losses['hour_of_day'] == worst_hour).mean()
                if pct_loss > 0.3:
                    insights.append(f"Appris de l'erreur : Les prises de position à l'heure {int(worst_hour)}h ont causé plusieurs pertes. Prudence accrue sur ce créneau.")

        # Analyser l'ATR (volatilité)
        if len(df) >= 10:
            high_atr_losses = df[(df['atr'] > df['atr'].median()) & (df['outcome'] == 0)]
            high_atr_wins = df[(df['atr'] > df['atr'].median()) & (df['outcome'] == 1)]
            if len(high_atr_losses) > len(high_atr_wins) * 1.5:
                insights.append("Appris de l'erreur : La forte volatilité (ATR élevé) dégrade la précision technique. Réduction des trades durant ces phases.")

        # Importance des caractéristiques si le modèle existe
        if self.model is not None:
            importances = self.model.feature_importances_
            best_feat_idx = np.argmax(importances)
            best_feat = self.feature_names[best_feat_idx]
            
            feat_translations = {
                "trend_slope": "la force de la tendance 24h",
                "trend_strength": "l'accélération de la tendance",
                "dist_support": "la distance par rapport au support 4h/2h",
                "dist_resistance": "la distance par rapport à la résistance 4h/2h",
                "is_order_block": "la présence d'un order block",
                "is_fib_zone": "le positionnement Fibonacci 0.5-0.618",
                "is_bos": "la cassure de structure (BoS)",
                "pattern_detected": "la détection de figures chartistes",
                "rsi": "le niveau RSI",
                "macd_hist": "l'histogramme MACD",
                "momentum": "l'impulsion du momentum",
                "hour_of_day": "l'heure de trading",
                "atr": "la volatilité (ATR)"
            }
            
            insights.append(f"Analyse IA : Le facteur le plus déterminant pour la réussite de nos trades est {feat_translations.get(best_feat, best_feat)}.")
            
        if not insights:
            insights.append("L'IA affine actuellement ses paramètres de filtrage basés sur les zones de retracement.")
            
        return insights
