# BTC Event Layer (research-only)

## Purpose

Add an Opta-like event/situation layer to BTC research without changing the
production predictor.

The first implementation normalizes existing free Binance USD-M Futures
observations into immutable events and derives causal situation snapshots from
those events.

## Current event types

- `kline_1m`: closed 1-minute Binance Futures candle plus taker-buy/sell split.
- `depth20`: bounded Binance partial-depth snapshot with top-of-book spread and
  five-level depth imbalance.

Each event stores:

- exchange/event time;
- `available_at_ms`;
- local retrieval time;
- source/venue/symbol;
- stable event ID;
- payload SHA-256;
- normalized payload.

## PIT rule

For a prediction at `prediction_time_ms`, an event is usable only when:

`event_time_ms <= prediction_time_ms`

and

`available_at_ms <= prediction_time_ms`.

Unknown or malformed availability is rejected in strict mode. The layer never
turns missing observations into zero-valued evidence.

## Derived situation card

The research-only card currently supports multiple windows (default 60 seconds
and 300 seconds) and exposes:

- event counts/types;
- total/buy/sell volume;
- taker imbalance;
- depth imbalance;
- spread;
- availability lag;
- freshness;
- exact event IDs used.

This makes the derived state traceable back to its source observations and
provides a foundation for later retrieval, similarity, regime, information-value,
and case-level studies.

## Deliberate non-adoption

This module is not wired into `predict.py`, the production feature schema, or
the promotion gate.

Adoption requires chronological OOS/holdout evidence, PIT validation,
calibration/robustness checks, and production-safe rollback evidence.
