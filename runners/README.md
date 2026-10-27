# Reference system runners

Each runner reads a build folder and writes one WAV per item, named by item id, to
`<build>/outputs/<system>/<item id>.wav`: 16 kHz mono 16-bit PCM, cut or zero-padded to the
mixture's length. The scorer reads the same folder:

    rightvoice score --build data/build/v1 --system <system>

The runners are scripts, not part of the `rightvoice` package, and import their model packages
only when they run, so `--help` works without them. Run one from the repository root:

    python runners/<runner>.py --build data/build/v1 [--quick] [--limit N] [--only clips|long] [--device D]

`--quick` runs only the quick subset that `rightvoice score --quick` scores, `--limit N` runs the
first N items, `--only clips` or `--only long` keeps the 30 s clips or the long files, and
`--no-resume` runs every item again. `--only long --limit 1` checks that a system copes with a
long file before the full run.

## The runners

| System | Script | Uses the voice sample | Licence of the model code |
| --- | --- | --- | --- |
| `control_clean`, `control_mixture`, `control_silence` | `controls.py` | no | not a model |
| `deepfilternet3` | `deepfilternet3.py` | no, by design | the `deepfilternet` package's own |
| `usef_tfgridnet` | `usef_tfgridnet.py` | yes | CC BY-NC 4.0, evaluation only |

**Controls.** Three systems with known answers, run before trusting any number: the clean
target stem, the mixture unchanged, and nothing at all. The first two are symbolic links to the
build's own files. See the docstring of `controls.py`.

**DeepFilterNet3.** A general speech enhancer that removes noise and reverberation and keeps
every voice. It ignores the voice sample by design, so it shows what plain enhancement does on
RightVoice: the partner's and the bystanders' words come through. It needs the `deepfilternet`
package (imported as `df`) and `torch`; `df.enhance.init_df()` loads DeepFilterNet3 and fetches
its weights on first use. The model works at 48 kHz, so the runner resamples to 48 kHz and back.
DeepFilterNet pins numpy below 2 and uses the torchaudio 2.0 API, so it runs in its own
environment; the runner needs only numpy, soundfile and soxr from this repository:

    uv venv .venv-deepfilternet --python 3.11
    uv pip install --python .venv-deepfilternet/bin/python "torch==2.0.1" "torchaudio==2.0.2" \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python .venv-deepfilternet/bin/python "numpy<2" soundfile soxr deepfilternet
    .venv-deepfilternet/bin/python runners/deepfilternet3.py --build data/build/v1 --device cpu

**USEF-TFGridNet.** Target speaker extraction from the paper's official code at
https://github.com/ZBang/USEF-TSE, with weights at https://huggingface.co/ZBang/USEF-TSE. The
code is licensed CC BY-NC 4.0, which forbids commercial use, so none of it is kept in this
repository and RightVoice only runs it to report its numbers. `--fetch` clones the code into
`<data_root>/third_party/usef_tfgridnet`; the weights come from the authors' Hugging Face
repository. The runner builds and calls the model the way the authors' `eval.py` does, with the
checkpoint published for noisy, reverberant mixtures by default; `--probe` prints what it would
use:

    python runners/usef_tfgridnet.py --fetch
    hf download ZBang/USEF-TSE --local-dir <data_root>/third_party/usef_tfgridnet/chkpt
    python runners/usef_tfgridnet.py --build data/build/v1

The model's sample rate comes from its config (8 kHz for the published checkpoints), so the
mixture and sample are resampled to it and the output back to 16 kHz. Files longer than
`--chunk-s` (30 s) run in overlapping windows, each with the whole sample. Its packages beyond
the scorer's (`python_speech_features`) are installed with `uv pip install`.

## Your own runner

`common.py` holds the loop the reference runners share, and it is a single file that needs only
numpy, soundfile and soxr, so it runs in your model's own environment. Copy it next to your code
and give it a function that takes the mixture, the voice sample (both 16 kHz mono float32) and
the item's manifest row, and returns the output at 16 kHz, or None to return nothing for that
item:

```python
import common  # common.py copied next to this file

def process(mixture, sample, row):
    return my_system(mixture, sample)

common.run_system("data/build/v1", "my_system", process, quick=True)  # the quick subset
common.run_system("data/build/v1", "my_system", process)              # then every item
```

Outputs land in `data/build/v1/outputs/my_system/`, which is where `rightvoice score --system
my_system` looks when `--outputs` is not given. A run resumes where it stopped, and an item that
raises is logged as an error without stopping the run. For a model that cannot hold a 10-minute
file at once, `common.in_chunks(fn, x, sr, chunk_s, overlap_s)` runs `fn` over overlapping
windows and joins them with a crossfade:

```python
def process(mixture, sample, row):
    return common.in_chunks(lambda x: my_system(x, sample), mixture, 16000,
                            chunk_s=30.0, overlap_s=2.0)
```

## The run log

Every item a run processes adds one JSON line to `<build>/outputs/<system>/runs.jsonl` with its
`id`, `status` (`done`, `abstained` or `error`), the `seconds` the system took on it, the item's
length in `audio_s` and the `error` text. Each run first adds a line with `"run": "start"` that
names the package, its version and the weights it loaded, so every number can be traced to the
model that made it. The log is appended to, never rewritten.

A run resumes by default: an item that already has an output file is skipped and gets no line.
Outputs are written through a temporary file, so a stopped run never leaves a partial file. An
error in one item is logged and the run goes on; the script exits with status 1 when any item
failed. An output louder than full scale is scaled down to a peak of 0.99 rather than clipped,
and the line records the gain in `gain_db`.

## Abstention

A system that returns nothing for an item abstains, and the runner writes no file for it. The
scorer treats a missing file as empty output, which is scored like silence: every target word
is lost, and nothing is said over silence or in a wrong voice. Abstaining is the right answer
on a target-absent item and costs every target word on any other item.

An item that failed also has no file, so it would be scored the same way. `runs.jsonl` is what
tells the two apart: `rightvoice score` counts a logged abstention as an answer, refuses a run
that answered fewer than 90% of the items, and names every item with neither an output nor a
logged abstention. Rerun the system until no item ends in `error` before scoring it (a resumed
run retries exactly the items without an output). A loop of your own that does not write
`runs.jsonl` returns nothing for an item by writing an empty WAV.
