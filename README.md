## Install

```
python3 setup.py install
```

## Demos

### Load and display waveform for speech

```
python3 examples/demo_waveform.py --wave examples/speeches.wav
```

### Load and display spectrogram for speech

```
python3 examples/demo_spectrogram.py --wave examples/speeches.wav
```

### Display room configuration

```
python3 examples/demo_room.py
```

### Generate and display room impulse responses

```
python3 examples/demo_reverb.py
```

### Perform MVDR beamforming

```
python3 examples/demo_mvdr.py --wave examples/speeches.wav
```

### Perform GEV beamforming

```
python3 examples/demo_gev.py --wave examples/speeches.wav
```
