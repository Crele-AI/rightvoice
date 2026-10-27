Made by [Crele](https://www.crele.co).

# RightVoice

RightVoice is a public benchmark for one job: given a recording of a conversation and a few
seconds of one person's voice, return only that person. It is the test to run when you need to
know which tools keep only the voice you want, and which ones let the other person through.

This repository holds everything needed to use and extend it: the code that builds the set from
public sources, the scorer and report, and runners for two open reference systems. The audio is
at https://huggingface.co/datasets/crele/rightvoice and the canonical page is
https://www.crele.co/benchmarks/rightvoice.

## What it measures

Every number is computed from the output audio alone, so any file-in, file-out system can be
scored, including closed products.

- **Target words lost.** Word error rate of the output against the target's reference text.
- **Wrong-voice words.** Words in the output that belong to another voice in the recording,
  reported per target word and per minute.
- **Talk over silence.** Words per minute and speech share while the target is silent, and on
  items where the target is absent altogether.
- **Target activity F1.** Speech activity of the output against the target's reference activity.
- **Voice match.** Speaker-embedding cosine between the output and the clean target.
- **Quality.** A non-intrusive MOS predictor on the output.
- SI-SDR is written to each item's scores for researchers and kept out of the headline table,
  because it measures waveform agreement with the clean stem rather than what a listener hears.

The report gives every system a 95% interval from a paired bootstrap over items. It calls one
system "clearly better" than another when they differ by at least 10% and the interval excludes
zero. It also breaks every number down by overlap, re-entry, level against the other voices,
bystanders, same-sex pairs, capture, voice sample length and device, and target-absent and long
items.

## The set

v1 holds 1,530 items, 16.45 hours at 16 kHz mono: 1,500 clips of 30 seconds and 30 files of 5 to
10 minutes. In 121 items the person in the voice sample never speaks, and the right answer is
silence. Two readers from the LibriTTS-R test and dev subsets hold a conversation on a turn-taking
timeline taken from the AMI meeting annotations, so gaps, overlaps, back-channels and long silences
before re-entry are real. The conversation is then captured the way phones and laptops capture
it: simulated rooms and a smaller set of measured reverberant rooms, FSD50K noise and music, a
bystander voice from a TV or phone speaker, automatic gain, clipping, Opus, AMR-NB and G.711 codecs,
and packet loss. Each item carries the mixture, the voice sample, the clean target stem, the other
stems, and a manifest row with text, word timings and bucket labels. Every item is a pure function
of a seed and an index.

Crele does not train on RightVoice. Its readers, rooms and noise clips stay out of every Crele
training set.

## Use it

Clone the repository and install the scorer. It needs Python 3.11 and
[uv](https://docs.astral.sh/uv/), and in practice a GPU (see "What scoring costs" below).

```sh
git clone https://github.com/Crele-AI/rightvoice && cd rightvoice
uv sync --extra score
uv run hf download crele/rightvoice --repo-type dataset --local-dir data/build/v1
```

The download holds the items, the manifest, and every item's scores for the reference systems
and the controls in `scores/`. You can also rebuild the set from the pinned public sources:

```sh
uv sync --extra build
uv run rightvoice sources plan          # lists every download with its size and URL
uv run rightvoice sources fetch all     # downloads and verifies the pinned sources
uv run rightvoice build --seed 20261022 --out data/build/v1
```

### Run your system

For every item, read `items/<id>/mixture.wav` and `items/<id>/sample.wav` (16 kHz mono) and
write `<id>.wav` into one folder. Any sample rate works: the scorer converts it and cuts or pads
it to the mixture's length. To return nothing for an item, as on a target-absent one, write an
empty WAV. Thirty items are long files of 5 to 10 minutes; if your model cannot take that much
audio at once, run them in overlapping windows.

Run your model in its own environment. `runners/common.py` is a single file that needs only
numpy, soundfile and soxr. Copy it next to your code and it handles the loop over the items,
resuming, windows for long files and a run log; `runners/README.md` shows how.

### Score it

Check a system on the quick subset first: 192 clips, every 8th clip of each capture condition
and of the target-absent clips, about a tenth of the set. The `items` command lists them, and
the helper's `--quick` runs only them.

```sh
uv run rightvoice items --build data/build/v1 --quick > quick.txt
uv run rightvoice score --build data/build/v1 --system my_system --outputs path/to/outputs --quick
uv run rightvoice report --build data/build/v1 --quick
```

Then run every item and score the full set:

```sh
uv run rightvoice score --build data/build/v1 --system my_system --outputs path/to/outputs
uv run rightvoice report --build data/build/v1
```

The report compares your system item by item with every system in `scores/`, the reference
systems and the controls included, with 95% intervals and the "clearly better" verdicts. The
quick report does the same on the quick subset. Quick numbers differ from the full set's by
several points, so use them to check a system and to compare it with others, never in place of
a full run.

Scoring refuses a system when fewer than 90% of the items have an output, so that a run that
stopped early is not scored as a system that returned nothing; `--allow-partial` scores it
anyway. Every item without an output is named, because it counts as empty output.

**What scoring costs.** Every output goes through Whisper large-v3, a voice activity detector,
a speaker encoder and a quality predictor. On one 24 GB GPU a full run, about 16 hours of
audio, took 2.5 to 5.5 hours per system and used about 8 GB of GPU memory; the quick subset
takes about a tenth of that. On a CPU it is many times slower. The first run downloads the
models, about 3 GB of them for Whisper, and `--asr-cache <folder>` keeps each recognition result
so that scoring the same outputs again skips it.

## Results

RightVoice v1, seed 20261022, 1,530 items. Lower is better for the first five columns, higher for
the last three.

| System | Target words lost | Wrong-voice words per 100 target words | Wrong-voice words per minute | Words per minute, target silent | Words per minute, target absent | Activity F1 | Voice match | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Crele Focus 1.1 | 33.9%\* | 3.7 | 2.98 | 11.2 | 2.0 | 0.834\* | 0.699 | 3.48 |
| USEF-TFGridNet | 43.3% | 24.5 | 22.76 | 48.1 | 65.9 | 0.799 | 0.570 | 2.90 |
| DeepFilterNet3 | 79.7% | 56.2 | 47.35 | 85.8 | 69.5 | 0.738 | 0.766 | 3.39 |

\* From a test version of Focus 1.1 with a different check for whether the sample's speaker is
present, set before RightVoice was run. Every other number is Focus 1.1 as released, which loses
39.4% of target words with an activity F1 of 0.782.

Positive controls, which check the scorer rather than a system:

| Control | Target words lost | Wrong-voice words per 100 target words | Words per minute, target silent | Words per minute, target absent | Activity F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| The clean target | 3.8% | 0.0 | 8.0 | 0.0 | 0.953 |
| The mixture, unchanged | 74.3% | 61.7 | 87.1 | 74.9 | 0.742 |
| Silence | 100.0% | 0.0 | 0.0 | 0.0 | 0.000 |

| System | What it is | How it was run |
| --- | --- | --- |
| Crele Focus 1.1 | Crele's speaker isolation model; closed | As released, except the two starred numbers. |
| USEF-TFGridNet | A published research model for voice-sample extraction (code CC BY-NC 4.0, evaluated only) | The authors' checkpoint for noisy, reverberant mixtures, which runs at 8 kHz; input and output resampled from and to 16 kHz. |
| DeepFilterNet3 | A general speech enhancer | Default settings. It ignores the voice sample and keeps every voice, which shows what cleaner audio alone does. |

How to read the table:

- **The controls set the floor.** A perfect output still reads 3.8% lost words, 8 words a minute
  while the target is silent (word times spill past the edges of each sentence) and an activity F1
  of 0.953. Read every system against those numbers, not against zero.
- **Empty outputs count.** A system that returns nothing loses every target word on that item and
  says nothing over silence. Voice match and quality are averaged over the outputs a system returns.
- **USEF-TFGridNet's quality and voice match** are lowered by its 8 kHz output, which carries nothing
  above 4 kHz.

## Licence

Code: Apache-2.0 (`LICENSE`). Audio, manifests and results: CC BY 4.0 (`DATA_LICENSE.md`), with
attribution to every source listed there.

## Cite

Cite it as "RightVoice (Crele)". `CITATION.cff` has the metadata.

```bibtex
@misc{rightvoice2026,
  title  = {RightVoice: a benchmark for keeping one voice from a conversation},
  author = {{Crele}},
  year   = {2026},
  url    = {https://www.crele.co/benchmarks/rightvoice}
}
```
