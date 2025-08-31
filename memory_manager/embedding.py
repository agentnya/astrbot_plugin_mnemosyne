import openai
from google import genai
from typing import List, Union
import os


class OpenAIEmbeddingAPI:
    """
    OpenAI compatible Embedding service wrapper class
    functionality：support specifying via environment variables or parameters API key
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None,
        base_url: str = None,
    ):
        """
        :param model: name of the embedding model used
        :param api_key: optional API key（takes precedence over environment variables）
        :param base_url: custom API endpoint（for compatibility with local deployment）
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        if not self.api_key:
            raise ValueError("must provide OpenAI API key or set OPENAI_API_KEY environment variable")

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def test_connection(self):
        # test withembedding connection
        try:
            response = self.client.embeddings.create(input=["hello"], model=self.model)
        except Exception as e:
            raise ConnectionError(
                f"Embedding Connection error: {e}\n please checkEmbedding whether the model configuration is correct，whether it is accessible"
            )

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        obtain text embedding vectors
        :param texts: input text（single string or list of strings）
        :return: list of embedding vectors
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            response = self.client.embeddings.create(input=texts, model=self.model)
        except Exception as e:
            raise ConnectionError(
                f"Embedding Connection error: {e}\n please checkEmbedding whether the model configuration is correct，whether it is accessible"
            )
        return [data.embedding for data in response.data]

class GeminiEmbeddingAPI:

    def __init__(self, model: str = "gemini-embedding-exp-03-07", api_key: str = None):
        """
        :param model: name of the embedding model used
        :param api_key: optional API key（takes precedence over environment variables）
        """
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("must provide Gemini API key or set GEMINI_API_KEY environment variable")

        self.client = genai.Client(api_key=self.api_key)

    def test_connection(self):
        # test with Gemini connection
        try:
            response = self.client.models.embed_content(
                model=self.model, contents="hello world"
            )
        except Exception as e:
            raise ConnectionError(
                f"Gemini Embedding Connection error: {e}\n please check whether the model configuration is correct，whether it is accessible"
            )

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        obtain text embedding vectors
        :param texts: input text（single string or list of strings）
        :return: list of embedding vectors
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            response = self.client.models.embed_content(model=self.model, contents=texts)
            embeddings = [embedding.values for embedding in response.embeddings]
            return embeddings

        except Exception as e:
            raise ConnectionError(
                f"Gemini Embedding Connection error: {e}\n please check whether the model configuration is correct，whether it is accessible"
            )