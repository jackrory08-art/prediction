"""Pluggable racing-data sources. Default: Betfair (free)."""
from .betfair import BetfairSource

__all__ = ["BetfairSource"]
