"""Decimal helpers for OHLCV and other money-like values (avoid float drift in APIs)."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any


def to_decimal_price(value: Any, exp: str = "0.0001") -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        q = Decimal(exp)
        return Decimal(str(value)).quantize(q, rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError, TypeError):
        return None


def to_volume_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))
    except (InvalidOperation, ValueError, TypeError):
        return None
