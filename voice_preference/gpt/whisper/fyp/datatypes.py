from pydantic import BaseModel, Field

class AudioParams(BaseModel):
    """Audio parameters for the whisper script"""
    format: int = Field(default= None, description="Audio format")
    channels: int = Field(default=1, description="Number of channels; 1 for mono, 2 for stereo")
    rate: int = Field(default=44100, description="Sampling rate")
    chunk: int = Field(default=1024, description="Chunk size")
    output_filename: str = Field(default="../audio_files/temp.wav", description="Output filename")