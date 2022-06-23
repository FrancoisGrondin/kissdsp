## Install

```
pip3 install -r requirements.txt
```

## Demos

### Load and display waveform for speech

```
python3 demo.py --operation waveform --in1 audio/speeches.wav
```

### Load and display spectrogram for speech

```
python3 demo.py --operation spectrogram --in1 audio/speeches.wav
```

### Display room configuration

```
python3 demo.py --operation room
```

### Generate and display room impulse responses

```
python3 demo.py --operation reverb
```

### Apply MVDR beamforming

```
python3 demo.py --operation mvdr --in1 audio/speeches.wav --out1 audio/noisy.wav --out2 audio/cleaned.wav
````

