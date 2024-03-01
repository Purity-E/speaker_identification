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
        #saving audio to memory
        waveform, sample_rate = torchaudio.load(audio)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        #applying pretrained pipeline
        dia = self.pipeline(audio_in_memory)
        assert isinstance(dia, Annotation)
        
        for i, value in enumerate(dia.itertracks(yield_label=True)):
            first_segment = value 
            break

        speech_turn = first_segment[0]
        turn = first_segment[1]
        current_speaker = first_segment[2]
        start = f"{speech_turn.start:.1f}"
        end_time = f"{speech_turn.end:.1f}"

        speakers = []
        starting = []
        ending = []

        for speech_turn, track, speaker in dia.itertracks(yield_label=True):
            start_turn = f"{speech_turn.start:.1f}"
            end_turn = f"{speech_turn.end:.1f}"
            if speaker != current_speaker and start_turn != end_turn:
                speakers.append(current_speaker)
                starting.append(float(start_time))
                ending.append(float(end_time))
                current_speaker = speaker
                start = f"{speech_turn.start:.1f}"
            else:
                start_time = start
                end_time = f"{speech_turn.end:.1f}"
                

        speakers.append(current_speaker)
        starting.append(float(start_time))
        ending.append(float(end_time))
                
        df = pd.DataFrame(list(zip(speakers, starting, ending)),
                    columns =['speakers', 'start', 'end'])
        df2 = df.drop_duplicates(subset=["start", "end"], keep='first')
        df2.reset_index(drop=True,inplace=True)
        def merge_consecutive_rows(df, condition_column):
            # Create a mask to identify consecutive rows
            mask = df[condition_column] != df[condition_column].shift(1)

            # Assign a group number to consecutive rows
            group_number = mask.cumsum()

            # Group by the consecutive groups and aggregate the data
            result_df = df.groupby([group_number], as_index=False).agg({
                condition_column: 'first',  # Take the first value in the group
                'start': 'first', 
                    'end' :'last'    # Example: sum other columns if needed
            })
            result_df['period'] = result_df['end'] - result_df['start']
            return result_df

        result_df = merge_consecutive_rows(df2, 'speakers')
    
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
    