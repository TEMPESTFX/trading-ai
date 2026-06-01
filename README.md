# IA de Trading Multi-Marchés (Forex, Exness MT5, Indices Synthétiques)

Ce projet implémente une Intelligence Artificielle (IA) de trading capable de négocier sur le Forex (ex: EURUSD, GBPUSD), les indices synthétiques de Deriv (ex: Volatility 75, Boom 500) et les marchés Weltrade. Elle est calibrée pour générer de micro-gains stables (ex: cibles de 0.50$ par transaction) et intègre un moteur d'apprentissage automatique qui retient ses erreurs de trading passées pour éviter de les reproduire.

## 🚀 Fonctionnalités Clés

1. **Analyse Technique Multi-Timeframe** :
   - **Tendance Globale (24h)** : Détectée par les pentes de moyennes mobiles exponentielles (EMA).
   - **Supports & Résistances (4h & 2h)** : Détection dynamique de zones clés par extraction d'extrema locaux (pics/creux).
   - **Fibonacci (Zones 0.5 et 0.618)** : Identification des impulsions et des niveaux de retracement idéaux pour l'entrée.
   - **Cassures & Retests (BoS - Break of Structure)** : Attente de la cassure de la structure de tendance et retest pour limiter le risque.
   - **Figures Chartistes** : Détection automatique des configurations classiques (Double Top, Double Bottom, etc.).

2. **Cerveau IA ("Learning Engine")** :
   - À chaque clôture de transaction (Gain ou Perte), l'IA enregistre les caractéristiques techniques (volatilité, heure, angle de tendance, Fibonacci, proximité S/R).
   - Un modèle de classification **Random Forest** est réentraîné dynamiquement.
   - Avant de prendre un nouveau trade, le modèle calcule la probabilité de réussite. Si elle est inférieure à **60%**, le trade est bloqué d'office pour éviter de reproduire une erreur passée.

3. **Gestion des Risques Précise** :
   - Calcule dynamiquement le lot en fonction de la distance du Stop Loss pour cibler exactement le montant de gain configuré (ex: 0.50$).
   - Limitation stricte de la perte pour empêcher les tirages massifs (drawdowns).

4. **Tableau de Bord Premium** :
   - Graphique de trading interactif en direct (propulsé par **TradingView Lightweight Charts**).
   - Visualisation en direct des statistiques de performance (gains totaux, taux de réussite, transactions bloquées par l'IA).
   - Liste des insights en langage naturel expliquant ce que l'IA a appris de ses échecs ("*Appris de l'erreur : Les trades à 18h ont un fort taux d'échec, prudence accrue.*").
   - Configuration facile en temps réel (cibles, mode simulation ou connexion MT5 réelle).

---

## 🛠️ Installation et Lancement

### 1. Prérequis
Assurez-vous d'avoir Python (3.10+) installé. Les bibliothèques requises sont :
```bash
pip install pandas numpy scikit-learn fastapi uvicorn requests websockets MetaTrader5
```
*(Note : La bibliothèque MetaTrader5 est facultative et s'exécute uniquement sous Windows si le terminal MT5 est installé. Si elle est absente, le bot fonctionnera automatiquement en mode Simulation autonome).*

### 2. Démarrer l'application
Depuis le dossier racine du projet (`trading_ai`), lancez la commande suivante :
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Accéder au Tableau de Bord
Ouvrez votre navigateur internet et naviguez sur :
```
http://localhost:8000
```

---

## 📂 Structure des Fichiers

- `app.py` : Serveur API FastAPI assurant la coordination, le service des fichiers web et la boucle d'analyse en arrière-plan.
- `strategy.py` : Module de calcul et d'identification des signaux techniques (EMA, S/R, Fibonacci, BoS Retest, Double Tops).
- `learning_engine.py` : Cerveau IA qui enregistre l'historique, entraîne la Forêt Aléatoire et prédit la viabilité des positions.
- `risk_manager.py` : Gestion du capital, calcul des tailles de lots et des objectifs de profit (0.50$) / pertes.
- `execution_engine.py` : Exécuteur d'ordres gérant la simulation de ticks en local et le pont vers le terminal MetaTrader 5 réel.
- `static/index.html` : Interface web haut de gamme (sombre, responsive et glassmorphic).
