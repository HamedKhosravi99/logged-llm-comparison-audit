import pandas as pd, pytest
from pathlib import Path
EX = Path(__file__).resolve().parents[1] / "examples"

@pytest.fixture
def log():      return pd.read_csv(EX / "routed_log.csv")
@pytest.fixture
def prices():   return pd.read_csv(EX / "model_prices.csv")
@pytest.fixture
def audit():    return pd.read_csv(EX / "randomized_audit.csv")
@pytest.fixture
def verified(): return pd.read_csv(EX / "verified_outcomes.csv")
@pytest.fixture
def traffic():  return pd.read_csv(EX / "traffic.csv")
