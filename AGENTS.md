# AGENTS.md

Guide for coding agents that call PostPrism or change this repository.

## Calling the service

- One route does the work: `POST /enrich`, multipart form data.
  - `text`: optional post text, up to 65,536 characters.
  - `files`: 0 to 10 repeatable fields, images or videos, in the post's order.
    Any other upload field name is rejected with HTTP 400.
  - A request needs text or at least one file.
- `GET /health` returns `status`, `service_commit`, `runtime_profile`, and the
  model identity of every worker. It returns 503 until the embedding model loads.
- No authentication. Call it from your backend over a private network, never from
  a browser.
- Rotate images upright before upload (for example Pillow's
  `ImageOps.exif_transpose`); the vector and caption ignore the EXIF Orientation tag.

## Using the response

- `multimodal_result.multimodal_embedding` is the 1024-number vector. Index it
  only when `multimodal_embedding_status` is `ok` or `text_only_ok`.
- Store the vector with every `multimodal_*` identity field. Never put vectors
  with a different model, revision, or method version into the same index.
- `visual_tag_result` and `caption_result` are `null` when those workers are off.
  Tags and captions are metadata: do not add them to the vector input.
- `asr_results` holds one entry per distinct video. Only entries with
  `asr_used_for_embedding: true` went into the vector; do not append transcripts
  to the vector input yourself.
- Timing maps are diagnostics and may gain keys; do not build logic on them.

Full contract: [docs/INTEGRATION.md](docs/INTEGRATION.md). Install:
[README.md](README.md). Artifacts: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Swapping to lighter models

If the owner asks for a lighter setup (see "Run it lighter" in
[README.md](README.md)):

- Captions, tags, and ASR are off unless `USE_CAPTIONS`, `USE_VISUAL_TAGS`, or
  `USE_ASR` is `true`. You don't need a code change to turn them off.
- The embedder has no config switch. Its identity is pinned in
  `postprism/multimodal/identity.py`, it is loaded by `load_qwen_embedding_adapter`
  in `postprism/multimodal/qwen_model.py` (called from
  `postprism/multimodal/embedder.py`), and `postprism/startup_checks.py` rejects
  any other model, revision, dtype, method version, or output size.
- A swap is a fork of the vector space. Update the identity fields and output size
  together, update the tests, and tell the owner that the database vector size has
  to match and that existing vectors must be re-embedded.
- `google/embeddinggemma-2` embeds text, images, video and audio in one 768-dim
  space (Apache-2.0, not yet tested here). It can replace both the embedder and
  the speech-to-vector step; captions and tags stay separate.
- `google/embeddinggemma-300m` is text only (768 dims). Pair it with an image
  model if images and video frames must stay searchable.

## Changing this repository

- Keep `POST /enrich` and `GET /health` as the only routes.
- Keep model inference, media decoding, and file work synchronous, and dispatch
  the complete enrichment call through `run_in_threadpool`.
- Keep the pinned Qwen model, revision, preprocessing, method version, dtype, and
  1024-number vector space. Changing any of them needs a new index and a full
  re-embed.
- ASR may enter the vector only when `asr_used_for_embedding is True`.
- The service must not queue work, write to a database, manage an ANN index, or
  accept model or checkpoint identifiers over HTTP.
- Never commit media, results, manifests, weights, caches, or secrets.
- Comment only non-obvious contracts, safety constraints, and calibrated constants.

Before committing, run:

```bash
./verify.sh
```
