"""
Author: Abraham R.

Description: 
This module serves as config reader for settings.yml file, containing secrets.

"""
from dataclasses import dataclass
import yaml


@dataclass
class RedisConfig:
    host: str 
    port: int 

@dataclass
class Ollama:
    url: str

@dataclass
class GLHF:
    url: str
    api_key : str = None


@dataclass
class Config:
    """
    Config dataclass maps settings.yml content into a Dataclass or a dictionary object.
    """
    glhf : GLHF = None
    ollama : Ollama = None
    redis: RedisConfig = None
    default_path: str = "settings.yml"


    @staticmethod
    def _read_file(filepath):
        """
        reads a file (protected)
        """
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
            return data
                
    @classmethod
    def from_yaml(cls, filepath: str = None):
        """
        reads settings from a yaml file given a filepath.
        """
        data = cls._read_file(filepath if filepath is not None else cls.default_path)
        glhf_data = data.get("GLHF", {})
        ollama_data = data.get("Ollama", {})
        redis_data = data.get("Redis", {})


        glhf_config = GLHF(
            url = glhf_data.get("Url", ""),
            api_key = glhf_data.get("ApiKey", "")
        )
        ollama_config = Ollama(
            url =  ollama_data.get("Url", ""),
        )
        redis_config = RedisConfig(
            host=redis_data.get("host", ""),
            port=redis_data.get("port",0)
        )
        return cls(
                glhf = glhf_config,
                ollama = ollama_config,
                redis = redis_config
                )

# check
if __name__ == "__main__":
    loader = Config().from_yaml()
    print(loader)