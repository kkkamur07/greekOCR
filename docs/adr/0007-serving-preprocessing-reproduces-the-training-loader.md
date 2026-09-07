# 0007. Serving preprocessing reproduces the training loader, exactly

- Status: Accepted
- Date: 2026-09-07
- Builds on: [0006](./0006-onnx-runtime-is-the-inference-runtime.md), which made the ONNX graph
  the run artifact and put the parity between trained and run model under test.

## Context

Pages that are *in* the training data did not come back from the platform as the ground truth
they were trained on, while the same models run on the training machine reproduced it. Same
weights, same page, different text. So the difference was not in the model.

It was in what the model was shown. The serving preprocessing under
`nomikos_inference/architectures/calamari/preprocessing/` was written against the legacy
TensorFlow Calamari processors, and it did what they do:

- centre-line dewarping (`center_normalize`, a per-column vertical shift estimated from the
  ink's centre of mass),
- inversion, so the line reaches the graph as white ink on a black ground,
- a 16 px zero pad on each side in `final_prepare_line_image`.

And the runner cut each line out of the page as a bare axis-aligned bounding box.

None of the registry models were trained that way. `armenian-calamari-v1`, `greek-calamari-v1`
and `syriac-calamari-v2` all come from the in-house PyTorch trainer,
`src/models/calamari/trainer.py`, whose loader `src/models/calamari/data.py::_load_line_image`
does three things and stops: `convert("L")`, an aspect-preserving BILINEAR resize to line height
48, a transpose to `time x height`. The `uint8` values are handed to the graph untouched, and
the graph divides by 255 itself. Dark ink on a light ground, no dewarp, no pad. The Armenian and
Syriac crops it read were written by
`src/preprocessing_data/syriac/xml_to_data.py::crop_polygon`: the polygon's bounding box widened
by 12 px, every pixel outside the polygon painted white, then `convert("L")`. The Greek ones came
through a different exporter and are tight boxes, which nobody noticed until the crop was
measured (see below).

So three of the operations on the serving path had no counterpart in training, and the graph was
being fed an image with the wrong polarity, the wrong geometry and the wrong margins on top of a
crop that matched no model's training data. The exporter had been stamping
`preprocessing: "existing Calamari NumPy preprocessing"` into the ONNX metadata the whole time,
which reads as a statement of fact and was never true of these models.

The mistake was not a bug in any one function. It was that "Calamari preprocessing" named a thing
in the wider Calamari world, and everybody assumed our models belonged to it, because nothing in
the repository ever compared the two paths.

## Decision

**The inference-time situation must exactly match the training situation.** Serving preprocessing
reproduces the training loader, and the runner's line crop reproduces the training crop writer.
Not approximately, not in spirit: the same PIL calls, the same rounding, the same rasteriser, the
same padding.

The **training recipe** of a model is two things, the loader and the crop convention, and only
the first turned out to be shared. All three sets came out of the same crop function, the
exporter's `crop_polygon`, but not with the same settings: the Greek finetuning crops were
exported with 0 px of padding where Armenian and Syriac used 12. A single padding cannot be right
for both, and the correct one is not derivable from the architecture or the script: it is a fact
about how that model's data was written, so it has to be recorded per model.

Concretely:

- `preprocessing/pipeline.py` is `_load_line_image` without the `torch.from_numpy`. Grayscale,
  aspect-preserving BILINEAR resize to the graph's line height, transpose, raw `uint8`. This half
  is shared by all three models.
- `preprocessing/crop.py` is new and holds `crop_line_on_white`, which is `crop_polygon` with a
  PIL page instead of a `cv2.imread` array. One crop function for every model: the polygon mask on
  a white canvas. `TRAINING_CROP_PADDING = 12` is the exporter's own `PADDING` and the default
  here, not a rule. It keeps `cv2.fillPoly` rather than switching to `ImageDraw.polygon`, because
  the two rasterise polygon edges differently and a one-pixel disagreement along every edge is the
  class of skew this whole record is about.
- **The runner re-encodes the crop as the exporter wrote it.** `save_crop` wrote every training
  line as a grayscale JPEG at `quality=82, optimize=True` and the trainer's loader read that file
  back, so the pixels the weights were fitted on had already been through a lossy encoder.
  `_crop_line_image` encodes the same way. A lossless PNG would hand the model a cleaner picture
  than any it was trained on, which is not the same thing as a correct one.
- **Coordinates are rounded the way the exporter's `parse_points` rounds them.** Supabase stores
  polygon points as floats with fractional parts on 92 of 819 Armenian and 116 of 204 Greek
  lines, and the exporter turned each into `int(round(float(x)))` before cropping. The first
  version of `crop.py` truncated instead, which shifts the box by one pixel on such lines and
  changed the text of 47 of them. `_integer_points` rounds, and the parity test feeds it the real
  `parse_points` output.
- **The decoder returns what the trainer's codec returns.** `_decode_greedy` used to trim leading
  and trailing whitespace after collapsing the CTC labels. The trainer's
  `src/models/calamari/codec.py::CharacterCodec.decode_ctc` does not, and its loader reads
  `.gt.txt` files without stripping, so a model that learned an edge space emits one on both
  sides. Over three whole training documents that trim was the last difference between the
  trainer's text and the runner's: 3 lines out of 1527.
- **The registry carries the crop settings.** Every `task: transcribe` entry has a required
  `line_crop` field, currently the single value `polygon-white` (the `crop_polygon` recipe, pixels
  outside the polygon painted white), and a required `line_crop_padding`: 0 for
  `greek-calamari-v1`, 12 for `armenian-calamari-v1` and `syriac-calamari-v2`. The runner reads
  both off the resolved registry entry and crops accordingly. They are required rather than
  defaulted, because a default is a guess about how somebody else's training data was written, and
  this record exists because such a guess went unexamined for three models.
- `preprocessing/geometry.py` (dewarping) and `preprocessing/final.py` (inversion and padding)
  are deleted. They were not disabled behind a flag: there is no model in the registry they are
  correct for, so a flag would only be a way to turn the defect back on. `line_crop_padding` is
  not the same kind of thing. It carries a number that is right for a real model, rather than
  keeping a wrong recipe reachable.
- `tests/inference/unit/test_calamari_training_parity.py` compares the serving functions
  byte-for-byte against the real training functions imported from `src/`. It is not a
  reimplementation of the recipe in test form; it calls both sides.

Registry pins are unchanged. The weights were always right; only the input was wrong.

## Measured

30 Armenian `MS_UCLA_MS` lines pulled from Supabase with their approved ground truth, one
`best.onnx` (`armenian-calamari-v1`, the pinned artifact), decoded both ways. The serving-path
column is byte-identical to what the cloud worker had already stored for those lines, so this is
the defect as users saw it and not a reconstruction of it.

| line preparation | exact lines / 30 | CER |
| --- | --- | --- |
| serving path as it stood (dewarp, invert, 16 px pad, unmasked box crop) | 1 | 0.52 |
| training recipe (`_load_line_image` + `crop_polygon`) | **30** | **0.00** |

Removing the three operations one at a time, on the same 30 lines, in the order they were peeled
off:

| removed | exact lines / 30 |
| --- | --- |
| inversion | 5 → 11 |
| centre-line dewarping | 11 → 27 |
| the 16 px pad | 27 → 30 |

Every one of the three mattered, and none of them alone was the whole story, which is why the
symptom read as "the model is worse than the training report says" rather than as a broken
pipeline.

Then the crop, which is where the two sets part company. Same loader and the same masked crop
function on both sides of each row, run over whole documents rather than a sample:
`armenian-calamari-v1` over the full `MS_UCLA_MS` document (819 lines) and `greek-calamari-v1`
over the whole `Grec1360` corpus (204 lines), geometry from Supabase, the same ONNX in every run:

| padding | Armenian (819 lines) | Greek (204 lines) |
| --- | --- | --- |
| 12 px | **664/819 exact** | 0/204 exact, CER 0.304 |
| 0 px | | **125/204 exact, CER 0.050** |

A bare unmasked box scored below both. So the mask is right for every model and the padding is
not: Greek gets every exact line it has from padding 0 and none at all from 12, which is the
difference between a usable transcription and an unusable one. The cause is provenance. Greek's
finetuning crops were exported with no padding, so "the training crop" is one function with two
settings, and reading Greek at Armenian's padding is the same category of error as reading either
with the TensorFlow processors. Hence `line_crop_padding` in the registry.

### Trainer versus serving, which is the benchmark

Ground truth is the wrong yardstick for a serving change: a model that never learned a line will
not reproduce it from any crop, and the Greek finetuning set has its own provenance gap (the
Supabase page is 1447 px wide where the training XML declares 1450). The yardstick is the trainer
itself. `train_side` here means the real `src/` code end to end: `crop_polygon` and `save_crop`
on the Supabase page, `CalamariLineDataset` reading the JPEG back, `best.pt` through the PyTorch
model, `CharacterCodec.decode_logits`. `serving` is the runner's `run_model` on the same page
bytes and the same polygons, through `best.onnx`.

| document | model | lines | serving == trainer | serving == ground truth |
| --- | --- | --- | --- | --- |
| Armenian `MS_UCLA_MS` | `armenian-calamari-v1` | 819 | **819** | 664 |
| Greek `Grec1360` | `greek-calamari-v1` | 204 | **204** | 53 |
| East Syriac `Chapter4` | `syriac-calamari-v2` | 504 | **504** | 0 |

Every line, byte for byte, with the trainer run at batch size 1. Three things had to be true for
that column to fill: the crop, the coordinate rounding, and the decoder above. Before the
rounding fix it read 813, 163 and 504; before the decoder fix 818, 202 and 504. `best.onnx`
against `best.pt` on the same tensors differs by at most `7e-5` in any logit and changes no
decoded line in any of the three documents.

**The batch size matters, and it is the trainer's doing.** Run at its default batch of 16 the
trainer disagrees with itself at batch 1 on 64 of the 504 Syriac lines. `collate_ctc` zero-pads
every line in a batch to the widest one, the convolution stack's "same" padding lets the last
valid frames see that pad (a `relu(bias)` plateau rather than background), and the backward
direction of the BiLSTM carries it across the whole line. So the text the trainer reports for a
line depends on which lines it shared a batch with, and its validation numbers are batch-layout
dependent. Serving runs one line per call, which is the trainer at batch 1, and that is the
correct one of the two: it is the only reading in which the model sees the line and nothing
else. The leak lives in `src/models/calamari`, outside this change, and is reported rather than
fixed here.

## Costs accepted

**Two trees now have to move together, and one of them is `src/`.** The training loader and the
crop writer live in the research tree, which is excluded from ruff and is not covered by the
inference test suite's normal reach. This record makes an edit there a change to serving
behaviour, which it did not used to be. The parity test is the whole of the enforcement.

**The parity test imports `src/`.** That is a dependency from `tests/inference/` into the research
tree, which nothing else there has, and it means the test cannot run inside the published-package
environment. Accepted: an oracle that is a copy of the thing it checks is not an oracle. This is
the same trade ADR 0006 made when it kept the export oracle in the repository rather than
restoring a `kraken`-based one.

**The ONNX `preprocessing` metadata string is now known to be wrong on artifacts already
published.** It says `existing Calamari NumPy preprocessing` on every Calamari graph in the
registry. Re-exporting to correct a comment would move three digests and three pins for no change
in behaviour, so it stands. Treat that field as a label somebody wrote once, not as an instruction
the runtime follows: the runtime follows this record.

**`line_crop` and `line_crop_padding` are required fields with no defaults, and a wrong value is
silent.** Registering a transcribe model is now two more things to get right, and getting the
padding wrong produces plausible text at six times the CER rather than an error. A default would
have hidden the Greek case entirely, which is the reason there is none; the honest mitigation is
that the value is measured against training pages before it is written, the way both values here
were.

**The Syriac padding is pinned to the exporter's constant, not measured.** `syriac-calamari-v2`
reproduces no ground truth we hold under any crop: on EastSyriac Chapter4 it reads 0/504 lines
exactly at a CER of about 0.29, unchanged by the padding. Whether that is a data-provenance
mismatch like Greek's page-width discrepancy or something about the model itself is not known
yet, so its `line_crop_padding: 12` is the exporter's own `PADDING` pending the training
manifests, and it is the one value in the registry that measurement has not confirmed.

**Edge whitespace now reaches the platform.** Two Greek lines end in a space the model learned
from its training text and the approved ground truth does not carry, so they count as ground-truth
mismatches (55 exact became 53) while being trainer-exact. If an edge trim belongs anywhere it is
in the comparison, not in the runtime, where it would be one more place the runtime silently
disagrees with the thing that produced the weights.

**Any model genuinely trained under the legacy processors can no longer be served.** There is no
such model in the registry, and if one arrives it needs its recipe carried with it rather than the
deleted code restored as a global mode.

## How to keep it true

A **training recipe** is a loader plus a crop convention, and both are recorded per model. These
files are one thing, and they change together:

| file | what it fixes |
| --- | --- |
| `src/models/calamari/data.py::_load_line_image` | how a line crop becomes a tensor |
| `src/preprocessing_data/syriac/xml_to_data.py::crop_polygon` | how a line crop is cut from a page (Armenian reaches it through `src/preprocessing_data/armenian.py`) |
| `src/preprocessing_data/syriac/xml_to_data.py::save_crop` | how that crop was written to disk, and so what the loader actually read: grayscale JPEG, `quality=82`, `optimize=True` |
| `src/preprocessing_data/syriac/xml_to_data.py::parse_points` | how float coordinates became the integers the crop was cut at: `int(round(x))`, not truncation |
| `src/models/calamari/codec.py::CharacterCodec.decode_ctc` | how labels became text: collapse repeats, drop blanks, return the rest untouched, edge whitespace included |
| `nomikos_inference/architectures/calamari/preprocessing/` | both of the above, at serving time |

`tests/inference/unit/test_calamari_training_parity.py` is what says so. It fails when the two
sides drift, which is the only mechanism here: reviewers did not catch this for the life of three
models, so the check has to be a test and not a convention.

A new `task: transcribe` entry cannot be registered without stating its `line_crop` and its
`line_crop_padding`. Both are required and neither has a default, so the question "how were this
model's training crops cut" has to be answered while the person who trained it is still in the
room, rather than inferred from a sibling entry later.

When a new architecture is added, the question to ask before writing its preprocessing is not
"what does this architecture usually want", it is "what did the trainer that produced these
weights actually feed the graph". Those are different questions, and this record exists because
the first one was asked. The Greek and Armenian paddings are the reminder that the answer can
differ between two models of the same architecture trained by the same team, on nothing more than
a flag somebody passed to an exporter.
