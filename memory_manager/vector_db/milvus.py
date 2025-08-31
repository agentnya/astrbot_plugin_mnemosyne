from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from typing import List, Dict, Optional, Any
from datetime import datetime
import time

from ...memory_manager.vector_db_base import VectorDatabase

from astrbot.core.log import LogManager


class MilvusDatabase(VectorDatabase):
    """
    Milvus Vector database implementation
    """

    def __init__(self, host, port):
        self.collections = {}  # Used to cache created collection instances
        self.connection_alias = "default"
        self.logger = LogManager.GetLogger(log_name="Mnemosyne MilvusDatabase")
        self.host = host
        self.port = port

    def __enter__(self):
        """Context manager：Connect to the database on entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager：Close the connection on exit"""
        self.close()

    def close(self):
        """
        Close database connection
        """
        try:
            if connections.has_connection(self.connection_alias):
                connections.disconnect(self.connection_alias)
                self.logger.info("and Milvus Disconnect.")
        except Exception as e:
            self.logger.error(f"and Milvus Error occurred during disconnection: {e}")

    def _ensure_connection(self):
        """Ensure connection with Milvus is valid，Reconnect if necessary"""
        if not connections.has_connection(self.connection_alias):
            self.logger.warning("and Milvus Connection has been disconnected，Attempting to reconnect...")
            self.connect()

    def connect(self):
        """
        Connect to Milvus Database
        """
        try:
            connections.connect(
                alias=self.connection_alias, host=self.host, port=self.port
            )
            self.logger.info(f"Successfully connected to Milvus Database ({self.host}:{self.port})")

            existing_collections = self.list_collections()
            self.collections.clear()  # Clear existing cache

            for col_name in existing_collections:
                try:
                    # Ensure collection exists
                    if not utility.has_collection(
                        col_name, using=self.connection_alias
                    ):
                        self.logger.debug(f"Collection '{col_name}' Does not exist")
                        continue

                    # Get collection object
                    col = Collection(col_name)

                    # Check if the collection has an index
                    if not col.indexes:
                        self.logger.warning(f"Collection '{col_name}' No index，Skip loading")
                        continue

                    # Load collection into memory
                    col.load()
                    self.logger.debug(f"Collection '{col_name}' Successfully loaded into memory")

                    # Cache collection object
                    self.collections[col_name] = col

                except Exception as e:
                    self.logger.error(f"Load collection '{col_name}' Failed: {e}")
                    continue  # If a collection fails to load，Continue to the next collection

        except Exception as e:
            self.logger.error(f"Connection Milvus Database failure: {e}")
            raise

    def _get_collection(self, collection_name: str) -> Collection:
        """Uniformly obtain collection instance and check load status"""
        col = self.collections.get(collection_name)
        if not col:
            if not utility.has_collection(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist.")
            col = Collection(collection_name)
            self.collections[collection_name] = col

        # Check if the collection is loaded
        load_state = utility.load_state(collection_name)
        if load_state != "Loaded":
            self.logger.info(
                f"Collection '{collection_name}' is not loaded. Loading now..."
            )
            col.load()  # Load collection
        return col

    def create_collection(self, collection_name: str, schema: Dict[str, Any]):
        """
        Create collection（Table）
        :param collection_name: Collection name
        :param schema: Field definition of the collection
        """
        try:
            # Check if the collection already exists
            if utility.has_collection(collection_name, using=self.connection_alias):
                self.logger.info(f"Collection '{collection_name}' Has been created.")
                return

            # Build field list
            fields = []
            for field_definition in schema["fields"]:
                field_name = field_definition["name"]
                field_type = field_definition["dtype"]
                is_primary = field_definition.get("is_primary", False)
                auto_id = field_definition.get("auto_id", False)
                is_nullable = field_definition.get("is_nullable", False)

                # Special handling：VARCHAR and FLOAT_VECTOR
                if field_type == DataType.VARCHAR:
                    max_length = field_definition["max_length"]
                    fields.append(
                        FieldSchema(
                            name=field_name,
                            dtype=field_type,
                            max_length=max_length,
                            is_primary=is_primary,
                            auto_id=auto_id,
                        )
                    )
                elif field_type == DataType.FLOAT_VECTOR:
                    dim = field_definition["dim"]
                    fields.append(
                        FieldSchema(name=field_name, dtype=field_type, dim=dim)
                    )
                else:
                    fields.append(
                        FieldSchema(
                            name=field_name,
                            dtype=field_type,
                            is_primary=is_primary,
                            auto_id=auto_id,
                            is_nullable=is_nullable,
                        )
                    )

            # Create collection's Schema
            collection_schema = CollectionSchema(
                fields, description=schema.get("description", "")
            )

            # Create collection
            self.collections[collection_name] = Collection(
                name=collection_name,
                schema=collection_schema,
                using=self.connection_alias,
            )
            self.logger.info(f"Collection '{collection_name}' Created successfully.")

            # Create index for vector field
            for field_definition in schema["fields"]:
                if field_definition["dtype"] == DataType.FLOAT_VECTOR:
                    index_params = field_definition.get("index_params", {})
                    if not index_params:
                        self.logger.warning(f"Index parameters not provided，Default use IVF_FLAT Index.")
                        index_params = {
                            "index_type": "IVF_FLAT",
                            "metric_type": "L2",
                            "params": {"nlist": 256},
                        }

                    # Create index
                    self.logger.info(
                        f"Creating index for field '{field_definition['name']}' Create index..."
                    )
                    self.collections[collection_name].create_index(
                        field_name=field_definition["name"], index_params=index_params
                    )
                    self.logger.info(
                        f"Field '{field_definition['name']}' Index created successfully."
                    )

            # Refresh collection to ensure data consistency
            self.collections[collection_name].flush()
            return

        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")

    def insert(self, collection_name: str, data: List[Dict[str, Any]]):
        try:
            self._ensure_connection()  # Ensure connection is valid
            col = self._get_collection(collection_name)

            # Automatically add timestamp
            current_time = int(time.time())
            for item in data:
                if "create_time" not in item:
                    item["create_time"] = current_time

            # Print debug information
            # self.logger.debug(f"Prepared for insertion field_data: {data}")

            col.insert(data[0])
            col.flush()  # Ensure data persistence
            self.logger.info(f"Data inserted successfully：{len(data)} .")

        except Exception as e:
            self.logger.error(f"Data insertion failed: {e}")

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
        try:
            self._ensure_connection()  # Ensure connection is valid
            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' Does not exist.")

            collection.load()

            results = collection.query(expr=filters, output_fields=output_fields)
            return results  # Directly return query results，query Results are usually iterable
        except Exception as e:
            self.logger.error(f"Conditional query failed: {e}")
            return []

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
        try:
            self._ensure_connection()
            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' Does not exist.")

            collection.load()

            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filters,
            )
            result_list = []
            for hits in results:  # Iterate over search results for each query vector
                for hit in hits:
                    # Replace here start
                    # Enhance type checking and error handling
                    if not all(
                        hasattr(hit, attr) for attr in ["id", "distance", "entity"]
                    ):
                        self.logger.warning(
                            f"Invalid search result object type: {type(hit)} | Content: {hit}"
                        )
                        continue

                    try:
                        entity_dict = (
                            hit.entity.to_dict()
                            if hasattr(hit.entity, "to_dict")
                            else dict(hit.entity)
                        )
                        result_list.append(
                            {
                                "id": hit.id,
                                "distance": hit.distance,
                                "entity": entity_dict,
                            }
                        )
                    except AttributeError as ae:
                        self.logger.error(
                            f"Entity parsing failed - Missing attribute: {ae} | data: {hit}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Unexpected error occurred while processing search results: {e} | data: {hit}"
                        )
                    # Replace here end
            return result_list
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []

    def list_collections(self) -> List[str]:
        """Get all collection names"""
        try:
            return utility.list_collections()
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []

    def get_loaded_collections(self) -> List[str]:
        """Get collections loaded into memory"""
        loaded = []
        for name in self.list_collections():
            col = Collection(name)
            if col.load_state == "Loaded":
                loaded.append(name)
        return loaded

    def get_latest_memory(
        self, collection_name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get the latest inserted memory"""
        try:
            # Use _get_collection Method ensures collection is loaded into memory
            collection = self._get_collection(collection_name)

            # Get latest records in descending order by timestamp（Correct sorting parameter format）
            results = collection.query(
                expr="",
                output_fields=["*"],
                sort_by=("create_time", "desc"),
                limit=limit,
            )

            # Safely handle empty results
            return results if results else []

        except ValueError as ve:
            self.logger.error(f"Collection does not exist: {ve}")
            return []

        except IndexError:
            self.logger.warning(f"Collection '{collection_name}' No data in")
            return []

        except Exception as e:
            self.logger.error(f"Failed to get latest memory: {e}")
            return []

    def delete(self, collection_name: str, expr: str):
        """Delete memory based on conditions"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist.")

            # Perform delete operation
            collection.delete(expr=expr)
            self.logger.info(f"Delete matching records: {expr}")
        except Exception as e:
            self.logger.error(f"Deletion failed: {e}")

    def drop_collection(self, collection_name: str) -> None:
        """
        Delete specified collection（Including all data under it）

        :param collection_name: Name of the collection to be deleted
        """
        try:
            if not utility.has_collection(collection_name):
                self.logger.warning(f"Attempt to delete non-existent collection '{collection_name}'")
                return

            # Unload collection from memory
            if collection_name in self.collections:
                self.collections[collection_name].release()

            # UsePymilvus APIDelete collection
            utility.drop_collection(collection_name)

            # Remove collection reference from local cache
            if collection_name in self.collections:
                del self.collections[collection_name]

            self.logger.info(f"Successfully deleted collection '{collection_name}' and all its data.")
        except Exception as e:
            self.logger.error(f"Error occurred while deleting collection: {e}")

    def check_collection_schema_consistency(
        self, collection_name: str, expected_schema: Dict[str, Any]
    ):
        """
        Check collection's Schema is consistent with expectations
        :param collection_name: Collection name
        :param expected_schema: Expected Schema Definition
        :return: True If consistent，False If inconsistent
        """
        try:
            self._ensure_connection()  # Ensure connection is valid

            # Check if collection exists
            if not utility.has_collection(collection_name, using=self.connection_alias):
                self.logger.warning(f"Collection '{collection_name}' Does not exist，Cannot check consistency.")
                return False

            # Get existing Schema
            collection = Collection(collection_name)
            existing_fields = {field.name: field for field in collection.schema.fields}

            # Boundary conditions：If existing fields are empty
            if not existing_fields and expected_schema["fields"]:
                self.logger.warning(
                    f"Collection '{collection_name}' No fields，Cannot match with expected Schema Match."
                )
                return False

            # Boundary conditions：If expected fields are empty
            if not expected_schema["fields"]:
                self.logger.warning(
                    f"Expected Schema Field definition is empty，Cannot match with collection '{collection_name}' Match."
                )
                return False

            # Extract field consistency check logic
            def check_field(
                field_definition: Dict[str, Any], existing_fields: Dict[str, Any]
            ) -> bool:
                field_name = field_definition["name"]
                field_dtype = field_definition["dtype"]

                # Check if field exists
                if field_name not in existing_fields:
                    self.logger.warning(
                        f"Collection '{collection_name}' Missing field '{field_name}'."
                    )
                    return False

                existing_field = existing_fields[field_name]

                # Check if field type is consistent
                if existing_field.dtype != field_dtype:
                    self.logger.warning(
                        f"Collection '{collection_name}' Field '{field_name}' Data type does not match. "
                        f"Expected: {field_dtype}, Actual: {existing_field.dtype}."
                    )
                    return False

                # Special handling：VARCHAR and FLOAT_VECTOR
                if field_dtype == DataType.VARCHAR:
                    expected_max_length = field_definition.get("max_length", None)
                    actual_max_length = getattr(existing_field, "max_length", None)
                    if expected_max_length != actual_max_length:
                        self.logger.warning(
                            f"Collection '{collection_name}' Field '{field_name}' Maximum length does not match. "
                            f"Expected: {expected_max_length}, Actual: {actual_max_length}."
                        )
                        return False
                elif field_dtype == DataType.FLOAT_VECTOR:
                    expected_dim = field_definition.get("dim", None)
                    actual_dim = existing_field.params.get("dim", None)
                    if expected_dim != actual_dim:
                        self.logger.warning(
                            f"Collection '{collection_name}' Field '{field_name}' Dimensions do not match. "
                            f"Expected: {expected_dim}, Actual: {actual_dim}."
                        )
                        return False
                return True

            # Iterate over expected field definitions
            for field_definition in expected_schema["fields"]:
                if not check_field(field_definition, existing_fields):
                    return False

            # Check for extra fields
            expected_field_names = {
                field["name"] for field in expected_schema["fields"]
            }
            extra_fields = set(existing_fields.keys()) - expected_field_names
            if extra_fields:
                self.logger.warning(
                    f"Collection '{collection_name}' Contains extra fields: {extra_fields}."
                )
                return False

            self.logger.info(f"Collection '{collection_name}' Structure is consistent with expectations.")
            return True

        except ConnectionError as ce:
            self.logger.error(f"Error occurred while connecting to the database: {ce}")
            return False
        except KeyError as ke:
            self.logger.error(f"Schema Missing key fields in definition: {ke}")
            return False
        except Exception as e:
            self.logger.error(f"Unknown error occurred while checking collection structure consistency: {e}")
            return False
