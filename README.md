# ***PMILx*** ***"Got Milx?"***

Audio-reactive polygon rendering in Python.

## ***DEMONSTRATION VIDEO***

![PMILX demo](demos/pmilx-demo.gif)

⚠️ Status: Active Development (MVP / Experimental)

---

## What is pmilx?

**pmilx** is an offline, audio-reactive visual renderer inspired by demoscene pipelines and classic MilkDrop aesthetics, built from scratch in Python.

Unlike preset-based visualizers, pmilx renders frame-by-frame with full control over geometry, camera transforms, trails, glow, and audio-driven structure. The goals of pmilx are to create deterministic, hackable visuals that can be re-rendered, mutated, and are intentional in their evolution.

**On the Name**

"pmilx" is a combination of Python, ProjectM, MilkDrop, and X for the experimental and hackable nature of the renderer.

Also, "pmilx" sounds silly and fun.

---

## Features (Current MVP)

- Offline frame-by-frame renders
- Audio band analysis (bass / mid / treble)
- Procedural polygon layers with lifetimes
- Deterministic camera motion with audio punch
- Temporal trails with controlled decay
- Bloom / glow post-process with clamped luminance
- Audio-triggered palette mutation
- Fully reproducible renders

---

## Philosophy

- Each frame starts empty and becomes real
- Effects are separated into clear passes
- Offline renders enable structures impossible in real-time engines

---

## Requirements

- Python 3.9+
- ffmpeg (installed and available in PATH)

Python dependencies:
- numpy
- librosa
- opencv-python
- scipy
- tqdm

---

## Setup

```bash
git clone https://github.com/hungry-bogart/pmilx.git
cd pmilx
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Place an audio file in the project directory titled audio/ at the top level.

Update AUDIO_FILE in render.py (see CONFIG section at the top of the script, example below.)

The audio file you use must match the name of the audio file set in the configuration section below. The audio filename the render script will access for visualizations needs to be in the /audio folder to work.

>The tests directory contains a preconfigured file.
>You can use that configuration if you just want to see
>what PMILx does. The main render script is for actual projects.

Configurations:

```
AUDIO_FILE = "./audio/example.mp3" # Sample file. See credits. Filename matches render.py
OUT_FILE = "./output/no-audio.mp4" # Pmilx generates a no audio file. If you only want the video.
FINAL_FILE = "./output/final-video-audio.mp4" # This is the rendered video with audio included.
```
Make sure:

- Your audio file matches`your actual filename.mp3`. The files in the configuration block are edited by hand for now. I realize this is not ideal. It's also an easy fix. I'll try to fix this, soon.

**WARNING: This means you may accidently overwrite files in the output directory if you don't change the values in the config block above.**

- Your no-audio output file is a silent video render. Named no-audio.mp4

- Final video muxed with audio via ffmpeg is named final-video-audio.mp4 or similar.

In sum, you can change the AUDIO_FILE, OUT_FILE, and FINAL_FILE variables in render.py to your own audio and output files. Just make sure they match what is provided in the configuration sample above.

Change the palletes. Adjust the glow values. Do whatever you want. Experiment with the script and add your own code. That's what it's all about.

## How to Run the pmilx Script

You can safely render your audio file now that you have edited the configuration.

Next, run the pmilx script:

1. Make it executable

```bash
chmod +x render.py
```

2. Run it

```bash
python3 render.py
```

> Note: The render may take a few minutes to complete, based on the length of your audio file. As a rule, try to use a shorter audio file for faster renders. This is important when you just want to try pmilx out for fun.

There may be times when it takes a longer time than expected, but you will always
see the progress bar. This is proof your render is working.

## Branch Strategy

> main: stable, tested development builds
> dev: active feature development and experiments

***Only tested features are merged into main.***

## Roadmap

 - [ ] Improved color energy preservation in glow

 - [ ] Multiple camera modes

 - [ ] Event-based structural mutations

 - [ ] Config-driven presets

 - [ ] Performance optimizations

 - [ ] Options flags so manual configs are no longer necessary

 - [ ] Config files to reproduce multiple outputs

---

### MIT License

Copyright (c) 2026 Hungry Bogart

---

### Credits

- [MilkDrop](https://github.com/milkdrop2077/MilkDrop3) - The original inspiration for this project.
- [Music](https://www.youtube.com/watch?v=jj2gHcSahKQ) - “Dredd - by Karl Casey @ White Bat Audio” Support them!
- [ProjectM](https://github.com/projectM-visualizer/projectm) - Another inspiration for this project and for its expansion of MilkDrop.
