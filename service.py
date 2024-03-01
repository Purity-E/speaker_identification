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
        #loading model into pipeline
        self.pipeline = Pipeline.from_pretrained("config.yaml")

        # sending pipeline to GPU (when available)
        self.pipeline.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
       
    @bentoml.api
    def diarize(self, audio: Path, context: bentoml.Context)-> t.Annotated[Path,bentoml.validators.ContentType('messages/*')]:
        import torchaudio
        from pyannote.core import Annotation
        import pandas as pd
        from pydub import AudioSegment
        from inference.inference import make_df
        from inference.inference import merge_consecutive_audios
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
        
        # merging consecutive audios from same speaker
        result_df = merge_consecutive_audios(df2, 'speakers')
    
        final_df = result_df[result_df['period']>=10]
        final_df.reset_index(inplace=True, drop=True)
        
        #creating directory
        directory = 'messages'
        directory_path = os.path.join(context.temp_dir,directory)
        os.mkdir(directory_path)
        
        #getting audio file
        audio_file = AudioSegment.from_mp3(audio)
        for i in range(0,len(final_df)):
            speaker_list = list(final_df.values[i,:])
            #cutting audio
            start_cut = (speaker_list[1])*1000
            end_cut = (speaker_list[2])*1000
            speech_file = audio_file[start_cut:end_cut]
            
            #saving file
            #output_path = os.path.join(context.temp_dir,f"{speaker_list[0]}.mp3")
            output_path = os.path.join(directory_path,f"{speaker_list[0]}-{i}.mp3")
            speech_file.export(output_path, format="mp3")
        #creating zip file
        zip_path = os.path.join(context.temp_dir,"messages.zip")
        #shutil.make_archive('E:/Zipped file', 'zip', directory_path)
        with zipfile.ZipFile(zip_path, 'w') as f:
            for file in glob.glob(f'{directory_path}/*'):
                f.write(file)
        return Path(zip_path)
    