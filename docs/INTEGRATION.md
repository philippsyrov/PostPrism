# Backend integration

This is the full `POST /enrich` and `GET /health` contract. The
[README](../README.md) has a short quickstart.

Your backend sends one complete post to `POST /enrich` as multipart form data:

- `text`: optional author text.
- `files`: zero to ten repeatable fields in the post's original order, images and
  videos in any mix.

At least one real text or media input is required. Media is accepted only through the
repeatable `files` field; a request that carries an upload under any other field name
(a legacy `file`, a typo) is rejected with HTTP 400 rather than embedded without that
media.

Send images already rotated upright. The embedding and caption workers use the stored
pixel order and ignore the EXIF Orientation tag, while RAM applies it, so a phone JPEG
that relies on that tag reaches the vector and caption sideways but the tags upright.
Apply the orientation to the pixels and drop the tag before upload (for example
Pillow's `ImageOps.exif_transpose`); an upright image without the tag reaches all
three workers identically. Changing this inside the service would change the
embedding method version.

## Response contract

`content_enrichment_status` is `ok` when the vector succeeds and every distinct
video's ASR finishes with `ok`, `no_speech`, `no_audio`, or `too_short`. It is
`partial` when the vector succeeds but ASR is disabled or fails for at least one
distinct video. It is `failed` when the canonical vector fails.

`multimodal_result` is always present:

| Field | Meaning |
| --- | --- |
| `multimodal_embedding` | 1024 floats; empty on failure |
| `embedding_dim` | `1024` |
| `multimodal_embedding_model` | model ID |
| `multimodal_embedding_model_version` | serving-policy version |
| `multimodal_embedding_model_revision` | immutable model commit |
| `multimodal_embedding_method_version` | preprocessing and pooling identity |
| `multimodal_embedded_at` | UTC timestamp |
| `multimodal_embedding_status` | `ok`, `text_only_ok`, `no_input`, `input_too_long`, `media_load_failed`, or `inference_failed` |
| `input_evidence` | ordered model-input provenance described below |

Only `ok` and `text_only_ok` contain an indexable vector.

`input_evidence` entries use `source_position` and `kind`. An image also has `sha256`;
a video frame has `fraction`, `seconds`, `frame_index`, and `sha256`; a duplicate has
`duplicate_of_position`; a decode failure uses `kind: media_load_failed`.

`asr_results` is always a list. It is empty for posts without videos and when ASR is
disabled; `media_summary.video_positions` tells those apart. Otherwise it holds one
entry per distinct video in media order. Byte-identical duplicate videos are
transcribed once at their first position and mapped through
`media_items.duplicate_of_position`. Each entry contains:

| Field | Meaning |
| --- | --- |
| `source_position` | index of the video in the submitted `files` order |
| `asr_transcript`, `language` | decoded speech and detected language |
| `duration_seconds`, `duration_after_vad`, `speech_ratio` | audio and VAD measurements |
| `duration_sent_to_asr` | seconds of audio handed to the ASR model after the 30 s activity cap; the transcript-density trust rules are judged against it |
| `asr_model`, `asr_model_version`, `asr_model_revision` | ASR identity |
| `asr_used_for_embedding` | literal per-video trust decision; only `true` entries entered Qwen input |
| `max_compression_ratio`, `asr_trust_reason` | transcript trust evidence |
| `audio_class`, `audio_class_confidence`, `audio_class_model`, `audio_class_revision`, `audio_class_coverage` | speech/music gate evidence |
| `asr_audio_selection` | `full-decode` for media up to 600 s; `timeline-probes` for longer media, where `duration_after_vad` and `speech_ratio` describe the probed audio rather than the whole timeline |
| `asr_transcribed_at` | UTC timestamp |
| `status` | `ok`, `no_speech`, `no_audio`, `too_short`, `audio_load_failed`, or `transcription_failed` |

Semantic topics are not part of this service. `USE_TOPICS` and
`TOPIC_DEFINITIONS_FILE` must be unset; either key fails startup, even when empty
or set to `false`.

When media is supplied, `media_items` contains `position`, `filename`, `sha256`, and
nullable `duplicate_of_position` for each submitted item. `media_summary` contains
`contract_version` (`post-media-v2`), submitted/distinct counts, `status`,
video/success/failure positions, and `duplicate_map`. The two timing maps are diagnostics and may gain new
keys; do not use them as business logic. `leg_timings_ms` separates `lock_wait`
(time queued behind the per-process inference lock), `asr` (audio decode, gating,
and transcription for every distinct video), `video_decode` (frame sampling),
`qwen_processing` (everything the embedding adapter does once frames exist:
chat-template assembly, visual preprocessing, processor tokenisation, device
transfer, the forward pass, and pooling), and `multimodal` (decode plus processing),
so a slow request can be attributed to queueing, media, or the model side.

Persist the vector together with every identity field in `multimodal_result`. Never
query one ANN index with vectors whose model, revision, method version, dtype,
preprocessing, or dimension differ. A future identity change requires a new index and
a full re-embed.

ASR is an internal ingredient, not a separate service. Only a literal
`asr_used_for_embedding: true` means the transcript was included in the Qwen input.
The backend may store `asr_results` for debugging, but it must not independently
append transcripts to the vector input.

## Video frame sampling

Every distinct video contributes at least one temporal pair of frames, and all video
frames in a post share one single-video-equivalent visual pixel and token budget,
keeping visual-token use near the single-video budget; the hard 8192-token
runtime check remains authoritative. One video keeps
eight proportional frames. Two videos receive four each, three receive
4 + 2 + 2 in post order, and four to ten receive one pair each (10% and 85%), so
five to ten videos carry 10 to 20 frames at proportionally smaller pixel sizes.
Four-frame allocations use 10%, 35%, 60%, and 85%. Short videos deduplicate repeated
frames and are never padded. Single-video and image-only requests are preprocessed
exactly as before, so existing vectors and the method identity stay valid;
`input_evidence` records the actual fractions, timestamps, and frame hashes. Frame
targets are measured from the stream's first presentation timestamp, so a clip whose
timestamps do not start at zero is still sampled across its whole length; this is the
`tsorigin` element of the method version and leaves zero-start media byte-identical.
`media_summary.failed_positions` names positions only when an item could not be
decoded; a Qwen context overflow fails the whole post without blaming any item.

Frames are located by seeking to the keyframe before each target timestamp and
decoding forward to the first frame at or after it, which selects the same frame,
timestamp, and pixels as the original sequential decode; containers that cannot
seek reliably fall back to that sequential decode. `frame_index` is the
presentation-order index; for a seek-decoded video it is the frame's position in
the sorted packet timestamp table read by demuxing the stream (no decoding), which
equals the sequential decode counter; a frame whose timestamp is missing from or
duplicated in that table forces the sequential fallback, as does any packet flagged
for discard or carrying a negative timestamp (the pre-roll left by a stream-copy
trim), so trimmed clips are decoded sequentially rather than indexed by a table the
decoder would not honour.

## Long-form ASR coverage

Media up to 600 s is decoded in full and its activity selected across the active
timeline. Longer media (roughly 15-minute videos within the 200 MB
per-video limit) is not decoded in full: the timeline is split into three equal
spans probed with seek-decoded 60 s chunks, at most 120 s per span. The first span
grows forward from the start of the media, the middle span grows outward from its
centre, and the last span grows backward from the very end, and each stops as soon
as it has found its 10 s share of the 30 s activity cap. Probes are planned over
elapsed duration, then translated to absolute decoder timestamps using the container
origin (missing or negative origins are clamped to zero). A genuine audio delay
behind video remains part of the overall media timeline, not a new zero point.
Each probe returns only audio
whose timestamps fall inside its window, placed at the timestamp it actually starts
from, so a stream that begins or ends inside a probe contributes a shorter span
rather than audio from elsewhere in the clip. Only the beginning, middle,
and end are sampled: speech that falls between the probes is not transcribed, and `asr_audio_selection: timeline-probes` marks
such rows. `duration_after_vad` counts all activity found in the probes while
`duration_sent_to_asr` is what reached the model; the density rules use the latter.
`asr_model_version` is now `qwen-v5`: unlike `qwen-v4`, probes account for the
container origin, so shifted-timestamp long clips can yield different transcripts;
zero-start selection is unchanged. Rows from the two versions are different ASR
extractions. `multimodal_embedding_method_version`
is the Qwen model-space identity (preprocessing, pooling, and vector space) and is
unchanged; vectors remain comparable across the ASR version bump because trusted
text enters the same model the same way. The per-video limit stays 200 MB (205 MB
per post); originals larger than that need a backend-generated analysis proxy (for
example a lower-bitrate transcode) before submission, because this service does not
transcode.

The backend owns queueing, retries, timeouts, post IDs, database writes, ANN indexing,
and ranking. One service process serialises its ASR and embedding model inference;
backend requests may wait for that per-process GPU lock. Size the backend request
timeout for the cumulative post: audio decode, AST gating, Qwen ASR, and frame decode
run sequentially for every distinct video, plus any lock wait, before the one
embedding.

With ASR, RAM++, and captions all enabled, the service used about 13 GB of GPU
memory on one NVIDIA A40. Latency depends on the GPU and on how many videos a post
has, so measure on your own GPU before sizing timeouts. As a starting point, not a
measured limit, keep at most one or two requests in flight per service process:
extra requests only wait on the lock, and each holds its uploaded media (up to
205 MB) in memory while it waits.

Recommended flow:

1. Queue the post after its text and ordered media are durable.
2. Call `/enrich` with the complete post.
3. Retry transport failures with the backend's normal idempotency policy.
4. Store successful or partial response fields against the backend's post ID.
5. Index only a successful canonical vector with the matching identity.

Use `/health` during deployment and reject a worker whose identity or
`runtime_profile` is not the expected production configuration. The `multimodal`
health block reports the model, model version, immutable revision, method version,
dtype, and embedding dimension. Set `SERVICE_COMMIT` to the deployed full 40-character
commit SHA so incidents can be traced to exact code.

## Optional visual metadata

`visual_tag_result` and `caption_result` are always present and are `null` when
their workers are disabled. Enabled text-only requests return `skipped_no_media`
and empty `items`. Otherwise each block reports `ok` when all inference items
succeed, `partial` when success and failure coexist, or `failed` when none succeed.
Items report `ok`, `media_load_failed`, `inference_failed`, or `skipped_duplicate`;
duplicates reference `duplicate_of_position` and do not count toward the rollup.
These statuses never alter `content_enrichment_status`, ASR, or an
indexable canonical vector.

Both blocks contain `model`, `model_revision`, `method_version`, `dtype`,
`generated_at`, and ordered `items`. Each item has `source_position`, `kind`, and
`status`; successful items record a hash of prepared RGB pixels. RAM additionally
reports checkpoint, vocabulary, and threshold SHA-256 identities. RAM uses
`RAM++-Swin-L-14M`, source `7cb804a8609e9f4b1a50b7f31436d2df40bb9481`, FP32,
EXIF-correct RGB, and the official 384 transform. Its
`ram-native-compact-display-v1` policy returns `native_tags` in original order and
at most six `display_tags` ordered by captured logits, with vocabulary order
breaking ties. The scores are uncalibrated and are not returned. Tags provide no
parent categories or moderation decisions; a missing tag is not evidence of absence.

RAM emits one item per image or sampled video frame, independently sampling at
most eight frames per distinct video (up to 80 across ten videos), with `fraction`,
`seconds`, and `frame_index`. There is no frame union or post-wide tag list.
This is separate from embedding's shared multi-video visual budget.

Captions use `Qwen/Qwen3-VL-2B-Instruct`, revision
`89644892e4d85e24eaac8bacfd4f463576704203`, BF16, deterministic promptv4 decoding,
512 output tokens, and one bounded retry for malformed output. Each distinct
media item gets one `caption`; a video seeks toward the midpoint and keeps the
first decoded frame, which may be a preceding keyframe. It records actual `seconds`
when available and `seek_fallback_used`. It describes that sampled view, not the
whole video. No OCR fields or carousel summary are returned. See each
`method_version` for the exact sampling/preprocessing identity; this clarification
does not rename the historical `middle-frame-v1` identity.

ASR and embedding run first, then RAM and captions run sequentially under the same
process inference lock. `leg_timings_ms` adds `visual_tags` and `captions` when
enabled. Cumulative latency and resident GPU memory must be measured with all
enabled workers, including mixed and ten-video posts; isolated model timings do
not establish service capacity. Optional per-item failures retain other results.

`/health` adds `visual_tags` and `captions`: disabled blocks contain `enabled: false`;
enabled blocks contain `enabled: true`, `available: true`, and fixed identities.
Enabled workers fail startup on dependency/artifact/warmup failure. RAM's checkpoint
and local tokenizer paths are operator-only environment settings, never HTTP
inputs. [Deployment provisioning](DEPLOYMENT.md)
covers the exact source patch, package/file hashes, local tokenizer provenance, and
unchanged runtime pins. Backend clients may persist this metadata alongside vectors,
but must not append captions or tags to vector inputs or alter the retrieval index
identity because optional metadata was enabled.
