# X research data policy

## Purpose

X/Twitter is a research-only external information source. It is **not** a production-model input in the current system.

## Accuracy rules

1. Preserve the source post text exactly; never replace it with an LLM summary.
2. Store immutable `post_id`, normalized author handle, canonical X URL, explicit UTC publication time, fetch time, and SHA-256 fingerprint.
3. Reject records whose publication time is after their fetch time.
4. Deduplicate only by immutable `post_id`.
5. A prediction at `decision_time_utc` may use only posts with `created_at_utc <= decision_time_utc`.
6. A post discovered later must not be backfilled into an earlier prediction snapshot.
7. Treat deleted, edited, unavailable, or ambiguously timestamped posts as unavailable rather than guessing.
8. Do not let X sentiment or narrative labels affect production predictions until a controlled OOS comparison demonstrates incremental value.
9. Compare X-augmented candidates against the unchanged production baseline on identical chronological OOS windows with purge/embargo where applicable.
10. Any schema, timestamp, identity, or provenance violation fails the X research audit closed.

## Source hierarchy

The X account supplied for this research (`@dravenip`) is currently treated as an external candidate source, not as a BTC signal. Its current public profile is primarily technology/AI-oriented, so BTC relevance must be established at the post level before any research feature is formed.

## Intended pipeline

`external fetcher -> raw snapshot -> strict normalization -> PIT audit -> research feature extraction -> chronological OOS evaluation -> candidate only`

The production predictor remains unchanged until evidence supports promotion.
