from __future__ import annotations
import bentoml
import typing as t
import shutil
import zipfile
import glob

from pathlib import Path
#from inference import inference
import os

#songornot_runner = bentoml.pytorch.get("songornot_model:latest", name = "songornot_runner", predict_fn_name="predict",).to_runner()
#svc = bentoml.Service(
    #runners=[
        #songornot_runner,
    #],
#)
#@svc.api(input=Path, output=Path)
#async def classifier(audio:Path, context: bentoml.Context)-> Path:
    #output = await songornot_runner.async_run(Path)
    #return output


@bentoml.service( resources={
        "gpu": 1,
        "memory": "8Gi",
    },
    traffic={"timeout": 900},
    )

class Diarization:
    def __init__(self) -> None:
        import torch
        from pyannote.audio import Pipeline
        from inference import config
        #loading model into pipeline
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=config.token)

        # sending pipeline to GPU (when available)
        self.pipeline.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
       
    @bentoml.api
    def diarize(self, audio: Path, context: bentoml.Context)-> t.Annotated[Path,bentoml.validators.ContentType('speakers/*')]:
        import torchaudio
        from pyannote.core import Annotation
        import pandas as pd
        from pydub import AudioSegment
        from inference.inference import make_df
        from inference.inference import merge_consecutive_audios
        import torch
        from inference.inference import create_embedding
        from inference.inference import get_similarity
        
        #saving audio to memory
        waveform, sample_rate = torchaudio.load(audio)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        #applying pretrained pipeline
        dia = self.pipeline(audio_in_memory)
        assert isinstance(dia, Annotation)
        # creating dataframe from diarization results
        df = make_df(dia.itertracks(yield_label=True))
        df2 = df.drop_duplicates(subset=["start", "end"], keep='first')
        df2.reset_index(drop=True,inplace=True)
        #dropping audios that are less than 10s
        df3 = df2[df2['period']>=10]
        df3.reset_index(inplace=True, drop=True)
        
        # merging consecutive audios from same speaker
        result_df = merge_consecutive_audios(df3, 'speakers')
        result_df.reset_index(inplace=True, drop=True)
        
        #creating directory for storing all cut audios
        directory = 'messages'
        directory_path = os.path.join(context.temp_dir,directory)
        os.mkdir(directory_path)
        #creating directory for certain speakers
        speaker_directory = 'speakers'
        speaker_path = os.path.join(context.temp_dir,speaker_directory)
        os.mkdir(speaker_path)
        
       #loading speaker embedding
        embedding_dict = torch.load('embedding_dict.pt')


        #cutting audios
        audio_file = AudioSegment.from_mp3(audio)
        for i in range(0,len(result_df)):
            speaker_list = list(result_df.values[i,:])
            #cutting audio
            start_cut = (speaker_list[1])*1000
            end_cut = (speaker_list[2])*1000
            speech_file = audio_file[start_cut:end_cut]
            audio_name = f"{speaker_list[0]}-{i}.mp3"
            
            #saving file cut audios
            output_path = os.path.join(directory_path,audio_name)
            speech_file.export(output_path, format="mp3")
            #create embedding, verifying audio & saving to speaker folder
            embedding = create_embedding(output_path)
            for key, value in embedding_dict.items():
                ans = get_similarity(embedding, value)
                if ans == 'yes':
                    audio_name = f"{key}-{i}.mp3"
                    speaker_output = os.path.join(speaker_path,audio_name)
                    speech_file.export(speaker_output, format="mp3")
                else:
                    continue
        #creating zip file
        zip_path = os.path.join(context.temp_dir,"speakers.zip")
        with zipfile.ZipFile(zip_path, 'w') as f:
            for file in glob.glob(f'{speaker_path}/*'):
                f.write(file)
        return Path(zip_path)
    