# Bbox correctness and downstream implications

PR #15's exact-parity FAIL remains unchanged. The
[canonical comparison](cold-start-20261030/bbox-correctness.md) and its JSON
retain all affected box indices, labels, coordinates, signed deltas and IoU.
The same comparison fields are retained for new candidate controls in the
[backend matrix](cold-start-20261030/candidate-backend-matrix.json).

The pinned renderer divides by **999**, then multiplies by the image dimension
and truncates to integer pixels. The separate Markdown image-reference converter
clamps at 1000. The fixtures' actual dimensions are 900×600 for text and
1000×650 for tables. A normalized delta of 3 is therefore about 2.70 horizontal
or 1.80 vertical pixels on a text fixture, before truncation. This is not a
universal pixel or OCR tolerance.

Across the ten historical pairs, eight fail exact output parity. All preserve
exact non-bbox bytes, page markers, canonical block/label/content order and
token count; differences are coordinate tokens. No reversed, zero-area or
out-of-range boxes are found. Minimum valid box IoU ranges down to
0.9253979567593252. An absolute coordinate bound alone hides relative changes
to narrow or short boxes.

A separate descriptive check uses only the frozen synthetic images. Every
non-white RGB pixel in each known generator text-row band contributes to its
ink extent; full table images contribute their table extent. For all 104
emitted boxes across both historical arms, the inclusive rectangle rendered
with the pinned scale contains the assigned measured ink extent. See
[pixel geometry](cold-start-20261030/synthetic-ink-geometry.json).
This detects no ink clipping under that disclosed interpretation. It is not
human annotation, a test of half-open downstream crop semantics, or an OCR
acceptance policy. The original images are hash-checked and never regenerated.

Reading order and text are unchanged in these outputs. A downstream system that
uses exact positions, containment, edge contact, overlap or spatial sorting can
still respond differently to moved edges; those algorithms were not exercised.
These measurements do not justify accepting every delta ≤3. The BF16 operator
0.03/0.03 criterion has no role in bbox acceptance. No relaxed semantic/bbox
policy is introduced or applied retroactively.
