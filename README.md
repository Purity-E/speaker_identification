# Speaker Identification
## Summary
The project is about extracting audios/speeches of different speakers from a recording while identifying the speaker. The outcome is audios/speeches of specific speakers in mp3 format named after the speaker.

## Environment Setup
- For pipenv environment run: pip install -r requirements.txt
- For conda environment run: conda install --file requirements.txt

## Steps involved 
## 1. Speaker Diarization 
- The diarization process is for getting which speaker spoke when.
- Used [pyannote diarization pipeline](https://huggingface.co/pyannote/speaker-diarization-3.1) from Hugging Face for diarization.
- It ingests mono audio sampled at 16kHz and outputs speaker diarization as an Annotation instance
- To convert audio to mono and sample rate of 16kHZ, run: ffmpeg -i "input.wav" -ar 16000 -ac 1 "output.wav"
## 2. Split the recording 
- Cut the audio into parts according to the diarization
- Used [pydub](https://github.com/jiaaro/pydub) to split the audio
## 3. Speaker Verification of the cut audio parts to get speaker identity
 - Used [pyannote model](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) to get embeddings for speaker verification
 - Run similarity of the split audio parts and the embeddings to verify the speakers.

## Model Deployment
Use [Bentoml](https://github.com/bentoml/BentoML/tree/main) for model deployment
## As a webservice
- Run: bentoml serve service:Diarization.
- Output is zipfile with audios of speakers
## Contenirize with docker
- To build a Bento: bentoml build
- To containerize bento: bentoml containerize diarization:q5k43xw4iwgsfcpo
- Run docker locally: docker run --rm -p 3000:3000 diarization:q5k43xw4iwgsfcpo
- Output is zipfile with audios of speakers


## Citations
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}

@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}