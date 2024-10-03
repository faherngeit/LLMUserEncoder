from typing import Literal
from functools import cache
import yaml
from openai_adapter import OpenAIAdapter

class EmbedAgent:
    """
    A class used to interact with the embedding agent.

    Attributes:
        agent (OpenAIAdapter): The adapter for the OpenAI API.

    Methods:
        get_user_description(item, test: bool = False): Gets the user description.
        encode_description(description: str): Encodes the description into embeddings.
        encode_user(user, test: bool = False): Encodes the user into embeddings.
    """

    def __init__(self, agent: Literal["openai"], model: str | None = None) -> None:
        """
        Initializes the EmbedAgent with the provided agent and model.

        Args:
            agent (Literal["openai"]): The agent to use for embeddings.
            model (str, optional): The model to use for generating embeddings. Defaults to None.
        """
        if agent == "openai":
            with open('tokens.yml', 'r') as file:
                token = yaml.safe_load(file)['openai']
            self.agent = OpenAIAdapter(token=token, model=model)

    def get_user_description(self, item, test: bool = False) -> str:
        """
        Gets the user description. This method should be implemented by the subclass.

        Args:
            item: The item to get the description for.
            test (bool, optional): Whether to run in test mode. Defaults to False.

        Returns:
            str: The description of the user.
        """
        pass

    @cache
    def encode_description(self, description: str) -> list[float]:
        """
        Encodes the description into embeddings.

        Args:
            description (str): The description to encode.

        Returns:
            list[float]: The embeddings of the description.
        """
        return self.agent.get_embedding(description)

    def encode_user(self, user, test: bool = False) -> list[float]:
        """
        Encodes the user into embeddings.

        Args:
            user: The user to encode.
            test (bool, optional): Whether to run in test mode. Defaults to False.

        Returns:
            list[float]: The embeddings of the user.
        """
        description = self.get_user_description(user, test)
        return self.encode_description(description)