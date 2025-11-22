# Kelly Clip Audio Inputs

Place the final ElevenLabs (or alternate VO) WAV files here so iClone can import them during AccuLips.

## Naming convention
```
clip01_warm_welcome.wav
clip01_warm_welcome_visemes.json
clip02_awe_and_wonder.wav
clip02_awe_and_wonder_visemes.json
...
```

- Audio must be **48 kHz mono WAV** for best AccuLips results.
- Viseme JSON should contain timestamp + phoneme arrays (any consistent schema is fine; the manifest references these files directly).

## Generating audio
1. Copy the text from `assets/kelly_clips/v1/scripts.json`.
2. Use `generate_lesson_audio_for_iclone.py` (or ElevenLabs UI) to export WAVs.
3. Drop the exports into this folder and keep file names identical to the clip IDs.

Once the WAV/viseme pairs are here, continue with the iClone animation workflow in `icl.plan.md`.








