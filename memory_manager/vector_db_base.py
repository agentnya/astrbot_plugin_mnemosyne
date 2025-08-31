from abc import ABC, abstractmethod
from typing import List, Dict, Any


class VectorDatabase(ABC):
    """
    Vector database base class
    """

    @abstractmethod
    def connect(self, **kwargs):
        """
        Connect to the database
        """
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, schema: Dict[str, Any]):
        """
        Create collection（Table）
        :param collection_name: Collection name
        :param schema: Field definition of the collection
        """
        pass

    @abstractmethod
    def insert(self, collection_name: str, data: List[Dict[str, Any]]):
        """
        Insert data
        :param collection_name: Collection name
        :param data: Data list，Each element is a dictionary
        """
        pass

    @abstractmethod
    def query(
        self, collection_name: str, filters: str, output_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Query data based on conditions
        :param collection_name: Collection name
        :param filters: Query condition expression
        :param output_fields: List of returned fields
        :return: Query results
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filters: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search
        :param collection_name: Collection name
        :param query_vector: Query vector
        :param top_k: Number of most similar results returned
        :param filters: Optional filtering conditions
        :return: Search results
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close database connection
        """
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """
        Get all collection names
        """
        pass

    @abstractmethod
    def get_loaded_collections(self) -> List[str]:
        """Get collections loaded into memory"""
        pass

    @abstractmethod
    def get_latest_memory(self, collection_name: str) -> Dict[str, Any]:
        """Get the latest inserted memory"""
        pass

    @abstractmethod
    def delete(self, collection_name: str, expr: str):
        """Delete memory based on conditions"""
        pass

    @abstractmethod
    def drop_collection(self, collection_name: str) -> None:
        """
        Delete specified collection（Including all data under it）

        :param collection_name: Name of the collection to be deleted
        """
        pass
