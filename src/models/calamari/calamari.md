# Calamari model package

This directory implements a Calamari-compatible, line-level OCR recognizer in
PyTorch. It is not the upstream `calamari-ocr` runtime. Instead, it provides
the pieces needed to train, fine-tune, evaluate, save, load, and export the
project's CNN–BiLSTM model.

`preprocessing/` is intentionally outside the scope of this document. The
files described here expect its output to be labelled line images and focus on
recognition rather than page or line-image preparation.

## What the package recognizes

The model performs **line OCR**: one input image corresponds to one complete
transcription string. It is not a page-layout model and does not predict line
coordinates.

At the package boundary, an image batch has NHWC layout:

```text
(batch, time/width, fixed_line_height, channels)
```

For the current grayscale pipeline, `channels` is `1`. The horizontal image
axis is called `time` because, after the CNN, it becomes the sequence consumed
by the BiLSTM and CTC classifier. Line widths may differ in a batch; `data.py`
pads them and preserves their original widths in `image_lengths`.

The recognition flow is:

```text
labelled line images
  -> CalamariLineDataset
  -> collate_ctc (right-pad images, concatenate targets)
  -> CalamariTorchModel
       CNN -> sequence features -> bidirectional LSTM -> class logits
  -> CTC loss during training / greedy CTC decoding during evaluation
  -> CER, WER, and exact-match metrics
```

The blank CTC class is always token ID `0`. All characters begin at ID `1`.
That invariant connects the codec, loss function, checkpoint metadata, and
ONNX metadata.

## `__init__.py`: public package boundary

`__init__.py` gives callers a compact public API instead of requiring imports
from implementation modules. It exports:

- `CalamariTorchModel` and the configuration types for model construction;
- `CharacterCodec` for character-to-token conversion;
- checkpoint save/load functions and `CalamariCheckpointMetadata`; and
- `export_calamari_onnx` for deployment export.

Training and evaluation helpers are deliberately not re-exported here, so
callers that need them import from `trainer.py` or `evaluate.py` explicitly.

The trailing docstring describes an intended architectural boundary: external
adapters may use the supported upstream Calamari distribution, while this
package owns this project's PyTorch implementation.

## `config.py`: architecture description and length arithmetic

This module holds immutable (`frozen`) dataclasses that describe the model
without building PyTorch layers yet.

### `CalamariTorchLayerConfig`

`CalamariTorchLayerConfig` is the schema for one architecture layer. Its
`kind` can be:

- `conv2d`: requires filters, kernel size, strides, padding, and optionally an
  activation;
- `maxpool2d`: requires pool size and strides;
- `bilstm`: requires `hidden_nodes` and currently only supports
  `merge_mode="concat"`; or
- `dropout`: uses `rate`.

Most fields are optional at the type level because each layer type needs a
different subset. `require_int` and `require_tuple` fail early with a useful
layer/field-specific error when a required value is missing.

### `CalamariTorchConfig`

`CalamariTorchConfig` contains the ordered layer tuple, the total number of
classes, and an optional output-logit temperature. Its
`downscaled_sequence_lengths()` method is critical for CTC correctness:

1. it receives original image widths;
2. it applies the horizontal stride of every convolution and pooling layer;
3. it uses ceiling division, matching TensorFlow-style `"same"` output-size
   behavior; and
4. it returns the number of valid output time steps for each image.

The model may receive a padded batch, but CTC must only see the valid
time-step count for each original image. Therefore `image_lengths` must be
passed to the model whenever batches contain variable-width lines.

`maxpool_strides()` treats a negative configured stride as "use the matching
pool size." The default topology uses `(-1, -1)` for both pools, which means
each is a 2×2, stride-2 pool.

### `default_model_config()`

This function builds the established recognizer topology:

```text
Conv2D(1 -> 40, 3×3, stride 1, same, ReLU)
MaxPool2D(2×2, stride 2, same)
Conv2D(40 -> 60, 3×3, stride 1, same, ReLU)
MaxPool2D(2×2, stride 2, same)
BiLSTM(200 units in each direction, concatenated)
Dropout(0.5)
Linear(400 -> number of CTC classes)
```

The two horizontal pooling operations reduce an input width to roughly one
quarter of its original size. The final class count is data-dependent: it is
the blank class plus every character in the training corpus.

## `layers.py`: PyTorch implementations of architecture primitives

This module turns the architecture descriptions in `config.py` into behavior
that matches the intended Calamari-style topology.

### `SameConv2d`

PyTorch's built-in convolution does not directly implement all desired
dynamic `"same"` padding behavior. `SameConv2d` calculates the needed padding
at runtime with `_pad_same()`, then applies `nn.Conv2d` with zero internal
padding. It accepts `same` and `valid` padding only. After convolution it
applies the configured activation.

Supported activations are ReLU, sigmoid, tanh, and leaky ReLU. Unknown
activation or padding names cause a `ValueError`, rather than silently
changing the network.

### `SameMaxPool2d`

This follows the same explicit-padding approach for pooling. It pads with
negative infinity rather than zero before max pooling. That matters at image
borders: zero could become a false maximum when genuine CNN feature values are
negative, while negative infinity cannot affect the maximum.

### `LazyBiLSTM`

The CNN's output feature width depends on the line height and preceding CNN
layers. `LazyBiLSTM` therefore creates its `nn.LSTM` on the first forward
pass, once the input feature width is known. It uses:

- `batch_first=True`, so sequence tensors are `(batch, time, features)`;
- `bidirectional=True`, so each time step sees both left and right context;
- the configured hidden size in each direction; and
- concatenation of both directional outputs, giving twice the configured
  feature count.

Lazy creation has an operational implication: the model must run once before
its parameters are passed to an optimizer or loaded from a checkpoint. The
trainer and checkpoint loader both explicitly materialize it before doing so.

### Tensor-shape helpers

`cnn_to_sequence()` transforms CNN output from NCHW:

```text
(batch, channels, time, height)
```

to a recurrent sequence:

```text
(batch, time, height * channels)
```

It keeps the horizontal axis as time and folds the remaining spatial height
and feature-channel dimensions into a feature vector.

`_same_padding_amount()` calculates the total padding that makes output size
equal to `ceil(input_size / stride)`. `_pad_same()` splits this padding as
evenly as possible across each side.

## `model.py`: CNN–BiLSTM recognizer and CTC logits

`CalamariTorchModel` is the `nn.Module` that assembles configured layers.
During construction it:

1. validates that there are at least two classes (blank plus one character);
2. converts each configuration entry into a CNN, pooling, BiLSTM, or dropout
   module;
3. tracks the CNN channel count needed by later convolutions; and
4. creates a lazy final linear classifier.

### `forward(image, image_lengths=None)`

The model expects a four-dimensional NHWC tensor. It rejects any other shape,
then converts pixel values from the expected `0–255` range to `0–1` float
values and permutes them to NCHW for PyTorch CNN layers.

The first `LazyBiLSTM` receives the CNN output after `cnn_to_sequence()`.
If a custom configuration contains no BiLSTM, the model still converts any
remaining four-dimensional CNN result into a sequence before classification.
The final lazy linear layer produces one logit per CTC class at each time
step.

The classifier's native ordering has blank last. `torch.roll(..., shifts=1)`
places that blank column at index `0` in the returned `logits`. This aligns
the model with `CharacterCodec`, `nn.CTCLoss(blank=0)`, and exported metadata.
`blank_last_logits` is also returned for consumers that need the original
classifier order.

When `temperature > 0`, the final logits are divided by that value. A value
greater than one softens the distribution; a value between zero and one
sharpens it. A non-positive setting disables scaling.

The returned mapping has:

- `blank_last_logits`: classifier output before blank reordering;
- `logits`: blank-first CTC logits, shape `(batch, output_time, classes)`; and
- `out_len`: valid output time steps after CNN/pooling downscaling.

## `codec.py`: deterministic CTC vocabulary and decoding

`CharacterCodec` owns the character-to-class mapping.

`from_texts()` collects every distinct Unicode character in the supplied
transcriptions, sorts them deterministically, and prepends the empty string
as the CTC blank. Sorting makes a newly trained model's mapping reproducible
for identical training text.

`__post_init__()` enforces three essential properties:

- there is a blank token at index `0`;
- the vocabulary contains at least one non-blank character; and
- no token occurs twice.

`encode()` converts a transcription into a one-dimensional `torch.long`
tensor of class IDs. A character absent from the codec is a hard error rather
than an unknown-token substitution. This is important for fine-tuning: the
existing classifier cannot represent unseen characters.

`decode_ctc()` implements greedy CTC collapse. Given the argmax token ID at
each time step, it removes blank tokens and adjacent repetitions. For example:

```text
token path:  0, α, α, 0, β, β, 0
decoded:     αβ
```

For repeated text characters, CTC requires a blank (or another token) between
them; the path `α, α` decodes to one `α`, while `α, 0, α` decodes to `αα`.

`decode_logits()` takes a batched logit tensor, computes the greedy argmax,
truncates each row to its valid `out_len`, moves it to CPU, and delegates to
`decode_ctc()`. It does not implement beam search or language-model scoring.

## `data.py`: labelled-line discovery, loading, and CTC collation

This module provides the dataset contract used for training and evaluation.
It does only recognition-side image loading and batching; it does not perform
the dedicated preprocessing pipeline.

### Supported dataset layouts

`collect_samples(root, split)` supports two formats:

1. A manifest layout with `gt_<split>.txt` and an `image/` directory. Each
   nonempty manifest row has `image_filename<TAB>transcription`.
2. A paired-file layout. For a split such as `train`, it first looks for a
   flat `root/train/` directory containing image files and matching
   `<stem>.gt.txt` labels. Otherwise it looks in `root/images/train/` and
   `root/labels/train/`.

Recognized image extensions are PNG, JPEG, TIFF, and their common variants.
Images without a paired label file are skipped.

A `source.txt` file can make the supplied root a virtual root. Its text is a
path to the actual dataset directory, either absolute or relative to
`source.txt`. This is useful for configuration directories that refer to
shared data without copying it.

### `CalamariLineDataset`

The dataset stores the discovered `LineSample` records. Every item:

1. opens the source image with Pillow;
2. converts it to grayscale;
3. rescales it to the fixed configured line height while preserving aspect
   ratio;
4. transposes it from image `(height, width)` to model `(width, height)`;
5. appends a one-channel axis, yielding `(time/width, height, 1)`; and
6. returns that tensor, encoded CTC target IDs, and the original text.

The loader leaves intensities as grayscale values. `model.py` performs the
`/255` normalization centrally, avoiding a training/inference discrepancy.

### `collate_ctc`

CTC supports variable sequence lengths, whereas dense tensors in a batch need
equal dimensions. `collate_ctc()` right-pads every line image to the widest
one with zeros, while preserving each original width as `image_lengths`.

For CTC loss, it produces a single concatenated target vector and matching
target lengths. It also produces zero-padded per-line `labels` for Hugging
Face Trainer's evaluation loop; blank is ID `0`, so those labels cannot be
confused with a real character target. The source `texts` are retained for
training-time metric logging.

## `trainer.py`: training, fine-tuning, validation, and checkpoint selection

`CalamariTrainingSettings` is the runtime configuration for a run. It
contains epoch and batching settings, optimizer hyperparameters, target line
height, device selection, temperature, optional checkpoint, mode, and names
of training/validation splits.

### `train_calamari`

`train_calamari(data_root, output_dir, settings, report=None)` is the main
training entry point.

For `mode="train"`, it gathers the training transcription texts, builds a
fresh deterministic codec from them, and creates the default architecture
with the corresponding class count.

For `mode="finetune"`, it requires `settings.checkpoint` and loads both the
model and persisted charset. Fine-tuning is rejected if:

- the requested line height differs from the checkpoint's height; or
- the new training texts contain any character absent from the saved codec.

These checks prevent loading weights into an incompatible input geometry or
classifier vocabulary.

The function uses Hugging Face's base `Trainer` with a CTC-specific adapter.
It materializes lazy layers before Trainer creates its optimizer, then uses
AdamW, cosine scheduling with the configured warmup ratio, gradient clipping,
and Trainer's standard device, mixed-precision, logging, and distributed
training lifecycle.

Trainer evaluates after every epoch and selects the checkpoint with the lowest
`eval_cer`. Each `checkpoint-N` contains Trainer's model, optimizer,
scheduler, RNG, and state files, plus Calamari's EMA weights and codec
metadata. Passing such a directory as `training.checkpoint` in `mode="train"`
resumes the complete training state after a Slurm time-limit cancellation.
`best.pt` remains a portable Calamari checkpoint for existing inference and
fine-tuning workflows.

Evaluation runs against EMA weights, while the raw model remains in the
Trainer checkpoint for an exact optimizer-state resume. The best EMA state is
used to write `best.pt`.

### Batch loss and evaluation helpers

`_batch_loss()` moves tensors to the chosen device, calls the model, applies
`log_softmax` across classes, and transposes logits to the
`(time, batch, classes)` layout required by PyTorch's `nn.CTCLoss`.
It also greedily decodes predictions with the codec. When no loss function is
provided, it returns a zero scalar so the same helper can support pure
inference-style evaluation.

`evaluate_model()` switches to evaluation mode and disables gradients. It
collects all decoded lines and references, calculates CER, WER, exact-match,
and SROIE precision/recall/F1, and adds mean CTC loss when a loss function was
requested.

## `src/metrics.py`: shared dependency-free text accuracy metrics

`edit_distance()` is a dynamic-programming Levenshtein distance
implementation. It compares arbitrary sequences, so it works for character
lists/strings and word lists alike. It holds only the previous and current
rows of the matrix, reducing memory use from a full two-dimensional matrix to
linear in prediction length.

`compute_text_metrics(references, predictions)` validates that corresponding
lists have equal length and returns:

- `cer`: total character edit distance divided by total reference characters;
- `wer`: total word edit distance divided by total reference words;
- `exact_match`: the fraction of lines whose prediction exactly equals the
  reference; and
- `sroie_precision`, `sroie_recall`, and `sroie_f1`: whitespace-token
  precision, recall, and F1 using the SROIE matching rule.

The denominators use `max(total, 1)`, so evaluating an empty reference set
does not cause division by zero. That keeps metrics numeric, although a
nonempty validation set is normally expected.

## `checkpoint.py`: safe, portable checkpoint persistence

This module defines the project checkpoint format `calamari-pytorch-v1`.

### Saved content

`save_calamari_checkpoint()` persists a plain dictionary containing:

- format identifier;
- class count;
- fixed line height;
- ordered blank-first charset;
- explicit `blank_index: 0`;
- temperature;
- the model `state_dict`.

It rejects a charset that does not reserve the first entry for the blank.
The checkpoint stores tensors and simple metadata, not a pickled model object,
which makes it portable and reduces the risk from executable deserialization.

### Loading and validation

`load_calamari_checkpoint()` calls `torch.load(..., weights_only=True)` on
CPU, checks the format identifier, validates all metadata and state-dictionary
value types, then recreates the default model architecture from the saved
class count and temperature.

Because the model contains lazy modules, it runs a small dummy image through
the model using the saved line height before calling `load_state_dict`.
Loading is strict: unexpected, missing, or shape-incompatible parameters
produce `CalamariCheckpointError`.

`CalamariCheckpointMetadata` is the validated metadata returned with the
ready-to-evaluate model. It preserves the information callers need to rebuild
the codec and input dataset correctly.

## `evaluate.py`: evaluate a saved checkpoint

`evaluate_checkpoint()` is a convenience wrapper for offline evaluation. It:

1. safely loads a checkpoint and its metadata;
2. rebuilds its codec from the saved charset;
3. loads the requested split at the checkpoint's required line height;
4. constructs a non-shuffled CTC data loader;
5. resolves `auto` to CUDA when available; and
6. calls `trainer.evaluate_model()` with CTC loss enabled.

It returns the same CER, WER, exact-match, SROIE, and mean-loss dictionary that
training evaluation uses. Keeping checkpoint evaluation on the shared
evaluation helper ensures that training-time and standalone metrics use the
same decoding and formulas.

## `export.py`: dynamic-width ONNX deployment export

`export_calamari_onnx()` turns a project checkpoint into an ONNX graph.
It loads the safe checkpoint, wraps the model so its dictionary output becomes
the two stable ONNX outputs `(logits, out_len)`, and exports with:

- inputs `image` and `image_lengths`;
- outputs `logits` and `out_len`;
- dynamic axis `1` for input width/time and output time; and
- default ONNX opset `17`.

The dummy export image has the saved line height, but dynamic axes allow
inference callers to supply arbitrary line widths. The exported graph still
expects NHWC grayscale images and an `image_lengths` tensor.

After export, the file is reopened with ONNX, its metadata is replaced, and
`onnx.checker.check_model()` validates the result before it is saved again.
The metadata describes the `calamari-onnx-v1` format, architecture, layout,
class count, line height, JSON-encoded Unicode charset, and blank index. This
lets a runtime decode output IDs without having to inspect the original
PyTorch checkpoint.

## Practical integration rules

When adding a caller or changing this package, preserve these contracts:

1. Keep line images NHWC with shape `(batch, width, line_height, 1)` at the
   model boundary.
2. Always provide true `image_lengths` for padded batches.
3. Keep the blank class at index `0` everywhere: codec, CTC loss,
   checkpoints, and exported metadata.
4. Materialize lazy modules before making an optimizer or loading weights.
5. Treat a checkpoint's charset and line height as immutable for fine-tuning
   and evaluation.
6. Use the checkpoint's metadata to configure inference, rather than assuming
   a particular alphabet or line height.
