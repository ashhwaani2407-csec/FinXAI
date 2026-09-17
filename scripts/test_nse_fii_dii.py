"""Regression test for NSE FII/DII parsing (nselib 2.4+ schema)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from backend.data_provider import MultiAssetDataProvider


class NseFiiDiiParseTest(unittest.TestCase):
    def test_parses_modern_nselib_schema(self):
        provider = MultiAssetDataProvider()
        sample = pd.DataFrame(
            [
                {"category": "DII", "netValue": 3908.23},
                {"category": "FII/FPI", "netValue": -2032.61},
            ]
        )
        with patch.object(provider, "_download_fii_dii_activity", return_value=sample):
            _, fii, dii, warnings = provider._fetch_nse_enrichment("RELIANCE")

        self.assertEqual(fii, -2032.61)
        self.assertEqual(dii, 3908.23)
        self.assertFalse(any("FII/DII data unavailable" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
