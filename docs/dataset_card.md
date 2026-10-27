---
license: cc-by-4.0
pretty_name: RightVoice
language:
  - en
tags:
  - audio
  - speech
  - target-speaker-extraction
  - speaker-isolation
  - benchmark
size_categories:
  - 1K<n<10K
---

# RightVoice

Made by [Crele](https://www.crele.co). The canonical page is https://www.crele.co/benchmarks/rightvoice and the code is at
https://github.com/Crele-AI/rightvoice.

RightVoice is a public benchmark for one job: given a recording of a conversation and a few seconds
of one person's voice, return only that person. It is the test to run when you need to know which
tools keep only the voice you want, and which ones let the other person through.

## What is in v1

1,530 items, 16.45 hours, 16 kHz mono 16-bit: 1,500 clips of 30 s and 30 files of 5 to 10 minutes.
121 items are target-absent: the person in the voice sample never speaks, and the right answer is
silence. Built from seed 20261022; every item is a pure function of the seed and its index.

- `manifest.jsonl`: one row per item with the conversation plan, readers, voice sample, bystanders,
  capture branch, room, background, device chain, levels and bucket labels.
- `items/<id>/`: `mixture.wav` (the recording), `sample.wav` (the voice sample), `target.wav` (the
  clean target), `partner.wav`, `others.wav` (mixture minus target minus partner).
- `selection.json`: the candidate pool and the quota each item filled.
- `attribution.csv`: every background clip used, with its uploader and licence.
- `build_info.json`: seed, counts, source digests, tool versions.
- `word_timings.json`: word times for every clean utterance used, from the recogniser the scorer
  uses; the scorer reads it so that wrong-voice words are matched by time.
- `listen.html`: players for 24 items.
- `scores/`: every item's scores for the two open reference systems (USEF-TFGridNet and
  DeepFilterNet3) and the three controls, so a report on your system compares with them item by
  item.

Two readers from the LibriTTS-R test and dev subsets hold a conversation on a turn-taking timeline
taken from the AMI meeting annotations, so gaps, overlaps, back-channels and long silences before
re-entry are real. The conversation is captured through simulated rooms and a smaller set of measured
reverberant rooms, FSD50K noise and music under CC0 or CC BY, a bystander voice from a TV or phone
speaker, automatic gain, clipping, Opus, AMR-NB and G.711 codecs, and packet loss.

## How to score a system

Write one 16 kHz WAV per item, named by item id, and score it with the code in the repository:

```sh
uv run rightvoice score --build <this dataset> --system my_system --outputs path/to/outputs
uv run rightvoice report --build <this dataset>
```

An empty file is scored as empty output, which is the right answer on a target-absent item. Add
`--quick` to both commands to check a system on 192 clips first. The repository's README covers
what scoring costs and how to run a system.

## Results

Reference results are in the repository's README. `rightvoice report` gives
any system you score 95% intervals and every bucket breakdown.

## Licence and attribution

CC BY 4.0. Attribute it as "RightVoice (Crele), https://www.crele.co/benchmarks/rightvoice", and keep
the attributions for its sources: LibriTTS-R (CC BY 4.0), the AMI Meeting Corpus annotations (CC BY
4.0), FSD50K clips under CC0 or CC BY (listed in `attribution.csv`), and BUT ReverbDB (CC BY 4.0).
`DATA_LICENSE.md` in the repository has the full citations.

Crele does not train on RightVoice. Its readers, rooms and noise clips stay out of every Crele
training set.

## Cite

```bibtex
@misc{rightvoice2026,
  title  = {RightVoice: a benchmark for keeping one voice from a conversation},
  author = {{Crele}},
  year   = {2026},
  url    = {https://www.crele.co/benchmarks/rightvoice}
}
```
