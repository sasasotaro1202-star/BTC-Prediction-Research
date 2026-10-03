# BTC Multi-Horizon Forecast Chart

docs/horizon_chart.html is a dependency-free chart viewer for the latest canonical multi-horizon forecast curve.

表示順は常に:

**5m → 10m → 15m → 30m → 1h → 3h → 6h → 12h → 24h**

5mを最優先表示とし、5m/10mはProduction Primary、15m以降はResearch Onlyです。

データは docs/horizon_curve_latest.json で、予測時点の確率とtarget時刻だけを使用します。未来のoutcomeはチャート生成に使用しません。

この表示追加はmodel/promotion policyを変更しません。