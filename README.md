Speaker Identification
## Summary
The project is about extracting audios/speeches of different speakers from a recording. For example having a recording from a meeting where different people spoke and extracting the speeches of the different speakers from the recording.
# steps involved 
- Speaker Diarization 
- Cutting the recording into parts according to the diarization
- Speaker Verification of the cut audio parts to get speaker identity

To test BentoML service run: 'bentoml serve service:Diarization'
To build a Bento 'bentoml build'
containerize bento 'bentoml containerize diarization:q5k43xw4iwgsfcpo'
Run docker locally 'docker run --rm -p 3000:3000 diarization:q5k43xw4iwgsfcpo'