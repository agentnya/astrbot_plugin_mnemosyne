import os
import pathlib
from typing import List, Dict, Optional, Any, Union
from urllib.parse import urlparse
import time

from pymilvus import connections, utility, CollectionSchema, DataType, Collection
from pymilvus.exceptions import (
    MilvusException,
    CollectionNotExistException,
    IndexNotExistException,
)
# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

from astrbot.core.log import LogManager

logger = LogManager.GetLogger(log_name="Mnemosyne")


class MilvusManager:
    """
    A class for managing interactions with Milvus a database。
    Encapsulates connection、Collection management、Data operations、Common functions such as indexing and searching。
    Supports connection to standard Milvus server (via URI or host/port) or use Milvus Lite (via local path)。

    Connection priority:
    1. If provided `lite_path`，then use Milvus Lite mode。
    2. If network is provided `uri` (http/https)，then use standard network connection。
    3. If explicitly provided `host` (non 'localhost')，then use host/port Connection。
    4. If none of the above is provided，then default to using Milvus Lite，Data path traces upwards from the current file4directory of the layer `milvus_data/default_milvus_lite.db`
    """

    def __init__(
        self,
        alias: str = "default",
        lite_path: Optional[str] = None,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[Union[str, int]] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        secure: Optional[bool] = None,
        token: Optional[str] = None,
        db_name: str = "default",
        **kwargs,
    ):
        """
        initialize MilvusManager。

        Args:
            # Parameter description remains consistent with before
            alias (str): Alias for this connection。
            lite_path (Optional[str]): Milvus Lite Local path of the data file。Prefer to use。
            uri (Optional[str]): standard Milvus Connection URI (http/https)。
            host (Optional[str]): Milvus Server hostname/IP。
            port (Optional[Union[str, int]]): Milvus Server port。
            user (Optional[str]): standard Milvus Authentication username。
            password (Optional[str]): standard Milvus Authentication password。
            secure (Optional[bool]): Whether for standard Milvus connection use TLS/SSL。
            token (Optional[str]): standard Milvus authentication Token/API Key。
            db_name (str): Name of the database to connect to (Milvus 2.2+)。
            **kwargs: Passed to connections.connect other parameters。
        """

        self.alias = alias
        self._original_lite_path = lite_path  # Retain original input for reference
        self._lite_path = (
            self._prepare_lite_path(lite_path) if lite_path is not None else None
        )

        self._uri = uri

        self._host = host
        self._raw_host = host
        self._port = str(port) if port is not None else "19530"

        self._user = user
        self._password = password
        self._secure = secure
        self._token = token
        self._db_name = db_name

        self.connect_kwargs = kwargs  # Store additional connection parameters

        self._connection_info = {}  # Used to store the final parameters passed to connect parameters
        self._is_connected = False
        self._is_lite = False  # Flag whether it is Lite mode

        # 3. Determine connection mode and configure parameters
        self._configure_connection_mode()

        # 4. Add general configuration (such as db_name)
        self._add_common_config()

        # 5. Merge additional kwargs parameters
        self._merge_kwargs()

        # 6. Attempt to establish initial connection
        self._attempt_initial_connect()

    # ------- Private method -------
    def _prepare_lite_path(self, path_input: str) -> str:
        """
        Prepare Milvus Lite path。If the input path does not end with .db end，
        then assume it is a directory，and append the default filename。
        Return absolute path。
        """
        path_input = os.path.normpath(path_input)  # Normalize path separators
        _, ext = os.path.splitext(path_input)

        final_path = path_input
        if ext.lower() != ".db":
            # Not ending with .db end，Assume it is a directory or basename，Append default filename
            final_path = os.path.join(path_input, "mnemosyne_lite.db")
            logger.info(
                f"Provided lite_path '{path_input}' Not ending with '.db' end。Assume it is a directory/basename，Automatically append default filename 'mnemosyne_lite.db'。"
            )

        # Ensure the return is an absolute path，Avoid confusion from relative paths
        absolute_path = os.path.abspath(final_path)
        logger.debug(f"Finally processed Milvus Lite absolute path: '{absolute_path}'")
        return absolute_path

    def _configure_connection_mode(self):
        """Decide connection mode based on input parameters and call the corresponding configuration method。"""
        # Note：Here self._lite_path is already processed _prepare_lite_path processed complete path
        if self._lite_path is not None:
            self._configure_lite_explicit()
        elif self._uri and urlparse(self._uri).scheme in ["http", "https"]:
            self._configure_uri()
        # Check host whether explicitly provided and not 'localhost' (case insensitive)
        elif self._host is not None and self._host.lower() != "localhost":
            self._configure_host_port()
        else:
            self._configure_lite_default()  # Default mode should also use _prepare_lite_path Calculate path

    def _ensure_db_dir_exists(self, db_path: str):
        """ensure Milvus Lite Directory where the database file is located exists。"""
        # db_path is already an absolute path
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):  # Check if directory exists
            try:
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"for Milvus Lite created directory: '{db_dir}'")
            except OSError as e:
                logger.error(
                    f"Unable to create for Milvus Lite create directory '{db_dir}': {e}。Please check permissions。"
                )
                # Perhaps an exception should be thrown here，because inability to create directory will cause connection failure
                # raise OSError(f"Unable to create for Milvus Lite create directory '{db_dir}': {e}") from e
            except Exception as e:  # Capture other potential errors
                logger.error(
                    f"Attempt for Milvus Lite create directory '{db_dir}' unexpected error occurred: {e}。"
                )
                # raise # Rethrow，Let the upper layer know an error occurred

    def _get_default_lite_path(self) -> str:
        """Calculate default Milvus Lite data path（Current file up4layer directory）。"""
        try:
            # Get definition MilvusManager absolute path of the class file
            current_file_path = pathlib.Path(__file__).resolve()
            # parents[0] is current directory, parents[1] is one layer up, ..., parents[3] is four layers up
            base_dir = current_file_path.parents[4]
            # Create a subdirectory in that directory to store data
            default_dir = base_dir / "mnemosyne_data"
            # Use _prepare_lite_path to ensure the final path is with .db file path
            default_path = self._prepare_lite_path(str(default_dir))
            logger.info(f"Dynamically calculated default Milvus Lite path is: '{default_path}'")
            return default_path
        except IndexError:
            # Use the default filename in the current working directory
            fallback_dir = "."
            fallback_path = self._prepare_lite_path(fallback_dir)
            logger.warning(
                f"Unable to get current file '{__file__}' up4layer directory structure，"
                f"Will use the current working directory's '{fallback_path}' as default Milvus Lite path。"
            )
            return fallback_path
        except Exception as e:
            fallback_dir = "."
            fallback_path = self._prepare_lite_path(fallback_dir)
            logger.error(
                f"Calculate default Milvus Lite unexpected error occurred while calculating path: {e}，"
                f"Will use the current working directory's '{fallback_path}' as default path。"
            )
            return fallback_path

    def _configure_lite_explicit(self):
        """Configure to use explicitly specified Milvus Lite path。"""
        self._is_lite = True
        # self._lite_path in __init__ already through _prepare_lite_path processing
        logger.info(
            f"configuration Milvus Lite (alias: {self.alias})。original input path: '{self._original_lite_path}', final data file path: '{self._lite_path}'"
        )

        # Ensure directory exists（Based on the final file path）
        self._ensure_db_dir_exists(self._lite_path)

        # Use the processed complete file path as URI
        self._connection_info["uri"] = self._lite_path
        logger.warning(
            "in Milvus Lite (explicit path) in mode，will ignore host, port, secure, user, password, token parameters。"
        )

    def _configure_lite_default(self):
        """Configure to use default Milvus Lite path。"""
        self._is_lite = True
        # call _get_default_lite_path Get the processed default file path
        default_lite_path = self._get_default_lite_path()
        logger.warning(
            f"No explicit connection method provided，will default to using Milvus Lite (alias: {self.alias})。data file path: '{default_lite_path}'"
        )

        # Ensure directory exists（Based on the final file path）
        self._ensure_db_dir_exists(default_lite_path)

        # Use the processed complete file path as URI
        self._connection_info["uri"] = default_lite_path
        logger.warning(
            "in default Milvus Lite in mode，will ignore host, port, secure, user, password, token parameters。"
        )

    def _configure_uri(self):
        """Configure to use standard network URI Connection。"""
        self._is_lite = False
        logger.info(f"Configure standard Milvus (alias: {self.alias}) Use URI: '{self._uri}'。")
        self._connection_info["uri"] = self._uri
        parsed_uri = urlparse(self._uri)

        # Handle authentication (Token Priority)
        if self._token:
            self._add_token_auth("URI")
        elif self._user and self._password:
            self._add_user_password_auth("URI")
        elif parsed_uri.username and parsed_uri.password:  # From URI Extract
            logger.info(f"From URI Extract from User/Password perform authentication (alias: {self.alias})。")
            self._connection_info["user"] = parsed_uri.username
            self._connection_info["password"] = parsed_uri.password

        # processing secure
        if self._secure is None:  # If not explicitly set
            self._secure = parsed_uri.scheme == "https"
            logger.info(
                f"According to URI scheme ('{parsed_uri.scheme}') infer secure={self._secure} (alias: {self.alias})。"
            )
        else:
            logger.info(
                f"Use explicitly set secure={self._secure} (URI Connection, alias: {self.alias})。"
            )
        self._connection_info["secure"] = self._secure

    def _configure_host_port(self):
        """Configure to use Host/Port connect standard Milvus。"""
        self._is_lite = False
        # host Already in _configure_connection_mode checked not to be None and non 'localhost'
        logger.info(
            f"Configure standard Milvus (alias: {self.alias}) Use Host: '{self._host}', Port: '{self._port}'。"
        )
        self._connection_info["host"] = self._host
        self._connection_info["port"] = self._port

        # Handle authentication (Token Priority)
        if self._token:
            self._add_token_auth("Host/Port")
        elif self._user and self._password:
            self._add_user_password_auth("Host/Port")

        # processing secure
        if self._secure is not None:
            self._connection_info["secure"] = self._secure
            logger.info(
                f"Use explicitly set secure={self._secure} (Host/Port Connection, alias: {self.alias})。"
            )
        else:
            self._connection_info["secure"] = False  # Default unsafe
            logger.info(
                f"Not set secure，Default to False (Host/Port Connection, alias: {self.alias})。"
            )

    def _add_token_auth(self, context: str):
        """Helper method：Add Token authentication information。"""
        if (
            hasattr(connections, "connect")
            and "token" in connections.connect.__code__.co_varnames
        ):
            logger.info(f"Use Token perform authentication ({context} Connection, alias: {self.alias})。")
            self._connection_info["token"] = self._token
        else:
            logger.warning(
                f"Current PyMilvus Version may not support Token authentication，will ignore Token parameters ({context} Connection)。"
            )

    def _add_user_password_auth(self, context: str):
        """Helper method：Add User/Password authentication information。"""
        logger.info(
            f"Use provided User/Password perform authentication ({context} Connection, alias: {self.alias})。"
        )
        self._connection_info["user"] = self._user
        self._connection_info["password"] = self._password

    def _add_common_config(self):
        """Add general configuration applicable to all connection modes，such as db_name。"""
        # processing db_name (Milvus 2.2+, to Lite and Standard All valid)
        if (
            hasattr(connections, "connect")
            and "db_name" in connections.connect.__code__.co_varnames
        ):
            if self._db_name != "default":
                logger.info(f"Will connect to the database '{self._db_name}' (alias: {self.alias})。")
                self._connection_info["db_name"] = self._db_name
            # else: No need to record using default library
        elif self._db_name != "default":
            mode_name = "Milvus Lite" if self._is_lite else "Standard Milvus"
            logger.warning(
                f"Current PyMilvus Version may not support multiple databases，will ignore db_name='{self._db_name}' (mode: {mode_name})。"
            )

        # Note：alias Do not put in _connection_info，It is connections.connect an independent parameter of

    def _merge_kwargs(self):
        """Merge additional user-provided kwargs parameters，Let explicit parameters take precedence。"""
        # final_kwargs = self.connect_kwargs.copy()
        # final_kwargs.update(self._connection_info) # let _connection_info settings take precedence
        # self._connection_info = final_kwargs
        # Change to： letkwargsSupplement，Do not overwrite already set parameters
        for key, value in self.connect_kwargs.items():
            if key not in self._connection_info:
                self._connection_info[key] = value
            else:
                logger.warning(
                    f"Ignore kwargs parameters in '{key}'，because it has been set by explicit parameters or internal logic。"
                )

    def _attempt_initial_connect(self):
        """Attempt to establish connection at initialization。"""
        try:
            self.connect()
        except Exception as e:
            mode = "Milvus Lite" if self._is_lite else "Standard Milvus"
            logger.error(
                f"Connect at initialization {mode} (alias: {self.alias}) Failed: {e}", exc_info=True
            )
            # Allow instance creation in case of connection failure，Subsequent operations will attempt to reconnect or report an error

    # ------- Public method -------
    def connect(self) -> None:
        """Establish to Milvus connection (According to the mode determined at initialization)。"""
        if self._is_connected:
            mode = "Milvus Lite" if self._is_lite else "Standard Milvus"
            logger.info(f"Connected to {mode} (alias: {self.alias})。")
            return

        mode = "Milvus Lite" if self._is_lite else "Standard Milvus"
        # From _connection_info Remove from alias，because it is to be used as connect the first parameter of
        connect_params = self._connection_info.copy()
        # connect_params.pop('alias', None) # pymilvus>2.4 Do not accept alias in kwargs

        logger.info(
            f"Attempt to connect to {mode} (alias: {self.alias}) Use parameters: {connect_params}"
        )
        try:
            # connections.connect(alias=self.alias, **connect_params) # Older pymilvus
            connections.connect(
                **{"alias": self.alias, **connect_params}
            )  # Works for pymilvus >= 2.4
            self._is_connected = True
            logger.info(f"Successfully connected to {mode} (alias: {self.alias})。")
        except MilvusException as e:
            logger.error(f"Connection {mode} (alias: {self.alias}) Failed: {e}")
            self._is_connected = False
            raise  # Retain original exception type
        except Exception as e:  # Capture other potential errors
            logger.error(f"Connection {mode} (alias: {self.alias}) Non-occurred at Milvus exception: {e}")
            self._is_connected = False
            # It may be better to wrap it as a more general connection error
            raise ConnectionError(f"Connection {mode} (alias: {self.alias}) Failed: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from Milvus server or Lite instance connection。"""
        if not self._is_connected:
            logger.info(f"Not yet connected to Milvus (alias: {self.alias})，no need to disconnect。")
            return
        mode = "Milvus Lite" if self._is_lite else "Standard Milvus"
        logger.info(f"Attempt to disconnect {mode} Connection (alias: {self.alias})。")
        try:
            connections.disconnect(self.alias)
            self._is_connected = False
            logger.info(f"Successfully disconnected {mode} Connection (alias: {self.alias})。")
        except MilvusException as e:
            logger.error(f"Disconnect {mode} Connection (alias: {self.alias}) Error occurred: {e}")
            self._is_connected = False  # Even if an error occurs，also marked as not connected
            raise
        except Exception as e:
            logger.error(f"Disconnect {mode} Connection (alias: {self.alias}) unexpected error occurred: {e}")
            self._is_connected = False
            raise

    def is_connected(self) -> bool:
        """Check current connection status (Use has_collection As ping)。"""
        if not self._is_connected:
            return False

        if self._is_lite:
            # Milvus Lite is a local file，Connection is usually more stable，Simply return the flag
            return True
        else:
            # For standard Milvus network connection，Perform a lightweight check
            try:
                # Use has_collection or list_collections
                utility.has_collection("__ping_test_collection__", using=self.alias)
                return True
            except MilvusException as e:
                logger.warning(
                    f"Standard Milvus Connection check failed (alias: {self.alias}): {e}"
                )
                self._is_connected = False
                return False
            except Exception as e:
                logger.warning(
                    f"Standard Milvus Unexpected error occurred during connection check (alias: {self.alias}): {e}"
                )
                self._is_connected = False
                return False

    def _ensure_connected(self):
        """Internal method，Ensure connected before performing operations。"""
        if not self.is_connected():
            mode = "Milvus Lite" if self._is_lite else "Standard Milvus"
            logger.warning(f"{mode} (alias: {self.alias}) Not connected。Attempt to reconnect...")
            try:
                self.connect()  # Attempt to reconnect
            except Exception as conn_err:
                # If reconnection fails，is_connected still is False
                logger.error(f"Reconnect {mode} (alias: {self.alias}) Failed: {conn_err}")
                raise ConnectionError(
                    f"Unable to connect to {mode} (alias: {self.alias})。Please check connection parameters and instance status。"
                ) from conn_err

        # Check again just in case connect() Internal logic issue
        if not self._is_connected:
            mode = "Milvus Lite" if self._is_lite else "Standard Milvus"
            raise ConnectionError(
                f"Failed to establish to {mode} (alias: {self.alias}) connection。Please check configuration。"
            )

    # --- Collection Management ---
    def has_collection(self, collection_name: str) -> bool:
        """Check if the specified collection exists。"""
        self._ensure_connected()
        try:
            return utility.has_collection(collection_name, using=self.alias)
        except MilvusException as e:
            logger.error(f"Check collection '{collection_name}' Error occurred when checking existence: {e}")
            return False  # or rethrow exception，depends on your error handling strategy

    def create_collection(
        self, collection_name: str, schema: CollectionSchema, **kwargs
    ) -> Optional[Collection]:
        """
        Create a new collection with the given schema。
        Args:
            collection_name (str): Name of the collection to be created。
            schema (CollectionSchema): Defines the collection structure CollectionSchema object。
            **kwargs: Passed to Collection Other parameters of the constructor (For example consistency_level="Strong")。
        Returns:
            Optional[Collection]: If successful，Then return Collection object，Otherwise Return None or existing collection handle。
        """
        self._ensure_connected()
        if self.has_collection(collection_name):
            logger.warning(f"Collection '{collection_name}' Already exists。")
            # Return handle of existing collection
            try:
                return Collection(name=collection_name, using=self.alias)
            except Exception as e:
                logger.error(f"Get existing collection '{collection_name}' Handle failed: {e}")
                return None

        logger.info(f"Attempt to create collection '{collection_name}'...")
        try:
            # Use Collection Class directly creates，It internally calls gRPC create
            collection = Collection(
                name=collection_name, schema=schema, using=self.alias, **kwargs
            )
            # Explicit call utility.flush([collection_name]) may help ensure collection metadata is updated
            # utility.flush([collection_name], using=self.alias)
            logger.info(f"Successfully sent create collection '{collection_name}' request。")
            return collection
        except MilvusException as e:
            logger.error(f"Create collection '{collection_name}' Failed: {e}")
            return None  # or throw exception
        except Exception as e:  # Capture other possible errors
            logger.error(f"Create collection '{collection_name}' unexpected error occurred: {e}")
            return None

    def drop_collection(
        self, collection_name: str, timeout: Optional[float] = None
    ) -> bool:
        """
        Delete specified collection。
        Args:
            collection_name (str): Name of the collection to be deleted。
            timeout (Optional[float]): Timeout for waiting for operation to complete（Seconds）。
        Returns:
            bool: Return if successfully deleted True，Otherwise Return False。
        """
        self._ensure_connected()
        if not self.has_collection(collection_name):
            logger.warning(f"Attempt to delete non-existent collection '{collection_name}'。")
            return True  # Can be considered target state achieved
        logger.info(f"Attempt to delete collection '{collection_name}'...")
        try:
            utility.drop_collection(collection_name, timeout=timeout, using=self.alias)
            logger.info(f"Successfully deleted collection '{collection_name}'。")
            return True
        except MilvusException as e:
            logger.error(f"Delete collection '{collection_name}' Failed: {e}")
            return False

    def list_collections(self) -> List[str]:
        """List Milvus all collections in instance。"""
        self._ensure_connected()
        try:
            return utility.list_collections(using=self.alias)
        except MilvusException as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        Get specified collection's Collection object handle。
        Args:
            collection_name (str): Collection name。
        Returns:
            Optional[Collection]: If collection exists，Then return Collection object，Otherwise Return None or throw exception。
        """
        self._ensure_connected()
        if not self.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' Does not exist。")
            # Can choose to throw CollectionNotExistException
            # raise CollectionNotExistException(f"Collection '{collection_name}' does not exist.")
            return None
        try:
            # Verify collection indeed exists and get handle
            collection = Collection(name=collection_name, using=self.alias)
            # Attempt to call a simple method to confirm handle validity，such as describe()
            # This will verify connection and collection existence
            # collection.describe()
            return collection
        except (
            CollectionNotExistException
        ):  # If has_collection and Collection State change between construction
            logger.warning(
                f"Get collection '{collection_name}' Handle found non-existent (May have just been deleted)。"
            )
            return None
        except MilvusException as e:
            logger.error(f"Get collection '{collection_name}' Error occurred when handling: {e}")
            return None
        except Exception as e:
            logger.error(f"Get collection '{collection_name}' Unexpected error occurred when handling: {e}")
            return None

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection statistics (e.g., number of entities)。
        Note：Getting accurate row count may require executing flush operation。
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return {"error": f"Collection '{collection_name}' not found."}
        try:
            # Ensure connection is valid
            self._ensure_connected()
            # First flush Get latest data
            self.flush([collection_name])  # Ensure statistics are relatively up-to-date
            stats = utility.get_collection_stats(
                collection_name=collection_name, using=self.alias
            )
            # stats Returns a dictionary containing 'row_count' keys such as
            row_count = int(stats.get("row_count", 0))  # Ensure it is an integer
            logger.info(f"Obtained collection '{collection_name}' statistics: {stats}")
            # Return standardized dictionary，Containsrow_count
            return {"row_count": row_count, **dict(stats)}
        except MilvusException as e:
            logger.error(f"Get collection '{collection_name}' Failed to get statistics: {e}")
            return {"error": str(e)}
        except Exception as e:  # Capture other errors
            logger.error(f"Get collection '{collection_name}' Unexpected error occurred during statistics: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

    # --- Data Operations ---
    def insert(
        self,
        collection_name: str,
        data: List[Union[List, Dict]],
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Optional[Any]:
        """
        Insert data into specified collection。
        Args:
            collection_name (str): Target collection name。
            data (List[Union[List, Dict]]): Data to be inserted。
                - If schema Fields ordered，Can be List[List]。
                - Recommended to use List[Dict]，Among them key is field name。
            partition_name (Optional[str]): Name of the partition to insert into。
            timeout (Optional[float]): Operation timeout。
            **kwargs: Passed to collection.insert other parameters。
        Returns:
            Optional[MutationResult]: Contains primary key of inserted entity (IDs) result object，Return if failed None。
        """
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Unable to get collection '{collection_name}' for insertion。")
            return None
        if not data:
            logger.warning(f"Attempt to insert into collection '{collection_name}' Insert empty data list。")
            return None  # or return an empty MutationResult
        logger.info(f"Into collection '{collection_name}' Insert {len(data)} data entries...")
        try:
            # ensure create_time Field exists and is INT64 timestamp
            current_timestamp = int(time.time())
            for item in data:
                if isinstance(item, dict) and "create_time" not in item:
                    item["create_time"] = current_timestamp
                # Add more checks if needed (e.g., for List[List])

            mutation_result = collection.insert(
                data=data, partition_name=partition_name, timeout=timeout, **kwargs
            )
            logger.info(
                f"Successfully inserted into collection '{collection_name}' Insert data。PKs: {mutation_result.primary_keys}"
            )
            # Consider whether to automatically flush，or let the caller decide
            # self.flush([collection_name])
            return mutation_result
        except MilvusException as e:
            logger.error(f"Into collection '{collection_name}' Data insertion failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Into collection '{collection_name}' Unexpected error occurred during data insertion: {e}")
            return None

    def delete(
        self,
        collection_name: str,
        expression: str,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Optional[Any]:
        """
        Delete entities in collection based on boolean expression。
        Args:
            collection_name (str): Target collection name。
            expression (str): Delete condition expression (For example, "id_field in [1, 2, 3]" or "age > 30")。
            partition_name (Optional[str]): Execute deletion in specified partition。
            timeout (Optional[float]): Operation timeout。
            **kwargs: Passed to collection.delete other parameters。
        Returns:
            Optional[MutationResult]: Contains primary key of deleted entity (If applicable) result object，Return if failed None。
        """
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Unable to get collection '{collection_name}' to execute deletion。")
            return None
        logger.info(
            f"Attempt to delete from collection '{collection_name}' delete entities meeting condition '{expression}' entities..."
        )
        try:
            mutation_result = collection.delete(
                expr=expression,
                partition_name=partition_name,
                timeout=timeout,
                **kwargs,
            )
            delete_count = (
                mutation_result.delete_count
                if hasattr(mutation_result, "delete_count")
                else "N/A"
            )
            logger.info(
                f"Successfully deleted from collection '{collection_name}' Send delete request。Number of deletions: {delete_count} (Note: Actual deletion requiresflushto take effect)"
            )
            # TODO Consider whether to automatically flush
            self.flush([collection_name])
            return mutation_result
        except MilvusException as e:
            logger.error(f"From collection '{collection_name}' Failed to delete entity: {e}")
            return None
        except Exception as e:
            logger.error(f"From collection '{collection_name}' Unexpected error occurred during entity deletion: {e}")
            return None

    def flush(self, collection_names: List[str], timeout: Optional[float] = None):
        """
        Persist insertions in memory of specified collection/Persist delete operations to disk storage。
        This is important for ensuring data visibility and accurate statistics。
        Args:
            collection_names (List[str]): List of collection names needing refresh。
            timeout (Optional[float]): Operation timeout。
        """
        self._ensure_connected()
        if not collection_names:
            logger.warning("Flush Operation requires specifying at least one collection name。")
            return
        logger.info(f"Attempt to refresh collection: {collection_names}...")

        try:
            for collection_name in collection_names:
                collection = Collection(collection_name, using=self.alias)
                collection.flush(timeout=timeout)
            # utility.flush(collection_names, timeout=timeout, using=self.alias)
            logger.info(f"Successfully refreshed collection: {collection_names}。")
        except MilvusException as e:
            logger.error(f"Refresh collection {collection_names} Failed: {e}")
            # Decide whether to throw exception as needed
        except Exception as e:
            logger.error(f"Refresh collection {collection_names} unexpected error occurred: {e}")
        return

    # --- Indexing ---
    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_params: Dict[str, Any],
        index_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """
        Create index on field of specified collection。
        Args:
            collection_name (str): Collection name。
            field_name (str): Field name to create index on (Usually a vector field)。
            index_params (Dict[str, Any]): Index parameter dictionary。
                Must contain 'metric_type' (e.g., 'L2', 'IP'), 'index_type' (e.g., 'IVF_FLAT', 'HNSW'),
                and 'params' (a dictionary containing index-specific parameters, e.g., {'nlist': 1024} or {'M': 16, 'efConstruction': 200})。
            index_name (Optional[str]): Custom name of the index。
            timeout (Optional[float]): Operation timeout。
            **kwargs: Passed to collection.create_index other parameters。
        Returns:
            bool: Return if index successfully created True，Otherwise Return False。
        """
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Unable to get collection '{collection_name}' to create index。")
            return False
        # Check if field exists in Schema in
        try:
            field_exists = any(f.name == field_name for f in collection.schema.fields)
            if not field_exists:
                logger.error(
                    f"Field '{field_name}' In collection '{collection_name}' of schema does not exist。"
                )
                return False
        except Exception as e:
            logger.warning(f"Error occurred when checking field existence（Collection may not be loaded or description failed）: {e}")
            # Continue attempting to create，let Milvus Decide

        # Build default index name
        default_index_name = (
            f"_{field_name}_idx"  # PyMilvus Default may use _default_idx or based on field
        )
        effective_index_name = index_name if index_name else default_index_name

        # Check if index already exists (Use provided name or possible default name)
        try:
            if collection.has_index(index_name=effective_index_name):
                logger.warning(
                    f"Collection '{collection_name}' field '{field_name}' already has a name '{effective_index_name}' index。"
                )
                return True  # Consider target achieved
            # If not specified index_name，also check if one already exists for that field index（Name may be unknown）
            elif not index_name and collection.has_index():
                # Further check if index is on target field
                indices = collection.indexes
                for index in indices:
                    if index.field_name == field_name:
                        logger.warning(
                            f"Collection '{collection_name}' field '{field_name}' Index already exists on (name: {index.index_name})。"
                        )
                        return True

        except MilvusException as e:
            # If has_index Error，Record and continue attempting to create
            logger.warning(f"Error occurred when checking index existence: {e}。Will continue attempting to create index。")
        except Exception as e:
            logger.warning(f"Unexpected error occurred when checking index existence: {e}。Will continue attempting to create index。")

        logger.info(
            f"Attempt to create index in collection '{collection_name}' field '{field_name}' on (name: {effective_index_name})..."
        )
        try:
            collection.create_index(
                field_name=field_name,
                index_params=index_params,
                index_name=effective_index_name,  # Use determined name
                timeout=timeout,
                **kwargs,
            )
            # Wait for index build to complete (Important!)
            logger.info("Wait for index build to complete...")
            collection.load()  # Loading is a prerequisite for searching，also implicitly trigger or wait for index
            utility.wait_for_index_building_complete(
                collection_name, index_name=effective_index_name, using=self.alias
            )
            logger.info(
                f"Successfully created and built index in collection '{collection_name}' field '{field_name}' on (name: {effective_index_name})。"
            )
            return True
        except MilvusException as e:
            logger.error(
                f"For collection '{collection_name}' Field '{field_name}' Failed to create index: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"For collection '{collection_name}' Field '{field_name}' Unexpected error occurred during index creation: {e}"
            )
            return False

    def has_index(self, collection_name: str, index_name: Optional[str] = None) -> bool:
        """Check if index exists on collection。"""
        collection = self.get_collection(collection_name)
        if not collection:
            return False
        try:
            return collection.has_index(index_name=index_name, timeout=None)
        except IndexNotExistException:  # Specifically capture exception of index non-existence
            return False
        except MilvusException as e:
            logger.error(
                f"Check collection '{collection_name}' index '{index_name or 'Any'}' Error occurred: {e}"
            )
            return False  # or throw exception
        except Exception as e:
            logger.error(
                f"Check collection '{collection_name}' index '{index_name or 'Any'}' unexpected error occurred: {e}"
            )
            return False

    def drop_index(
        self,
        collection_name: str,
        field_name: Optional[str] = None,
        index_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Delete index on collection。Prefer to use index_name，If not provided, attempt based on field_name Delete（May delete default index on that field）。"""
        collection = self.get_collection(collection_name)
        if not collection:
            return False

        effective_index_name = index_name  # Prefer to use explicit name

        # If not provided index_name，Attempt to find associated with field_name index
        if not effective_index_name and field_name:
            try:
                indices = collection.indexes
                found = False
                for index in indices:
                    if index.field_name == field_name:
                        effective_index_name = index.index_name
                        logger.info(
                            f"Found with field '{field_name}' index: '{effective_index_name}'。"
                        )
                        found = True
                        break
                if not found:
                    logger.warning(
                        f"In collection '{collection_name}' Not found with field in '{field_name}' index，Unable to delete。"
                    )
                    return True  # No corresponding index，Consider target achieved
            except Exception as e:
                logger.error(
                    f"Find field '{field_name}' Error occurred when finding index: {e}。Unable to continue deletion。"
                )
                return False
        elif not effective_index_name and not field_name:
            logger.error("must provide index_name or field_name to delete index。")
            return False

        # Check if index exists
        try:
            if not collection.has_index(index_name=effective_index_name):
                logger.warning(
                    f"Attempt to delete non-existent index（name: {effective_index_name}）in collection '{collection_name}'。"
                )
                return True  # Consider target state achieved
        except IndexNotExistException:
            logger.warning(
                f"Attempt to delete non-existent index（name: {effective_index_name}）in collection '{collection_name}'。"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Check index '{effective_index_name}' Error occurred when checking existence: {e}。Will continue attempting to delete。"
            )

        logger.info(
            f"Attempt to delete collection '{collection_name}' index on (name: {effective_index_name})..."
        )
        try:
            collection.drop_index(index_name=effective_index_name, timeout=timeout)
            logger.info(
                f"Successfully deleted collection '{collection_name}' index on (name: {effective_index_name})。"
            )
            return True
        except MilvusException as e:
            logger.error(
                f"Delete collection '{collection_name}' index on '{effective_index_name}' Failed: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Delete collection '{collection_name}' index on '{effective_index_name}' unexpected error occurred: {e}"
            )
            return False

    # --- Search & Query ---
    def load_collection(
        self,
        collection_name: str,
        replica_number: int = 1,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """
        Load collection into memory for searching。
        Args:
            collection_name (str): Name of collection to load。
            replica_number (int): Number of replicas to load。
            timeout (Optional[float]): Operation timeout。
            **kwargs: Passed to collection.load other parameters。
        Returns:
            bool: Return if successfully loaded True，Otherwise Return False。
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return False
        # Check load status
        try:
            progress = utility.loading_progress(collection_name, using=self.alias)
            # progress['loading_progress'] Will be 0 to 100 integer，or None
            if progress and progress.get("loading_progress") == 100:
                logger.info(f"Collection '{collection_name}' Loaded。")
                return True
        except Exception as e:
            if e.code == 101:  # Collection not loaded
                logger.warning(f"Collection '{collection_name}' Not yet loaded，Will attempt to load。")
            else:
                logger.error(
                    f"Check collection '{collection_name}' Error occurred when checking load status: {e}。Will attempt to load。"
                )

        logger.info(f"Attempt to load collection '{collection_name}' into memory...")
        try:
            collection.load(replica_number=replica_number, timeout=timeout, **kwargs)
            # Check load progress/Wait for completion
            logger.info(f"Wait for collection '{collection_name}' to load...")
            utility.wait_for_loading_complete(
                collection_name, using=self.alias, timeout=timeout
            )
            logger.info(f"Successfully loaded collection '{collection_name}' into memory。")
            return True
        except MilvusException as e:
            logger.error(f"Load collection '{collection_name}' Failed: {e}")
            # Common error：Index not created
            if "index not found" in str(e).lower():
                logger.error(
                    f"Load failure reason may be collection '{collection_name}' Index not yet created。"
                )
            return False
        except Exception as e:
            logger.error(f"Load collection '{collection_name}' unexpected error occurred: {e}")
            return False

    def release_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> bool:
        """Release collection from memory。"""
        collection = self.get_collection(collection_name)
        if not collection:
            return False

        # Check load status，No need to release if not loaded
        try:
            progress = utility.loading_progress(collection_name, using=self.alias)
            if progress and progress.get("loading_progress") == 0:
                logger.info(f"Collection '{collection_name}' Not loaded，No need to release。")
                return True
        except Exception as e:
            logger.warning(
                f"Check collection '{collection_name}' Error occurred when checking load status: {e}。Will attempt to release。"
            )

        logger.info(f"Attempt to release collection from memory '{collection_name}'...")
        try:
            collection.release(timeout=timeout, **kwargs)
            logger.info(f"Successfully released collection from memory '{collection_name}'。")
            return True
        except MilvusException as e:
            logger.error(f"Release collection '{collection_name}' Failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Release collection '{collection_name}' unexpected error occurred: {e}")
            return False

    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        vector_field: str,
        search_params: Dict[str, Any],
        limit: int,
        expression: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Optional[List[Any]]:  # Return type is List[SearchResult]
        """
        Perform vector similarity search in collection。
        Args:
            collection_name (str): Name of collection to search。
            query_vectors (List[List[float]]): Query vector list。
            vector_field (str): Name of vector field to search。
            search_params (Dict[str, Any]): Search parameters。
                Must contain 'metric_type' (e.g., 'L2', 'IP') and 'params' (a dictionary containing search-specific parameters, e.g., {'nprobe': 10, 'ef': 100})。
            limit (int): Number of most similar results returned for each query vector (top_k)。
            expression (Optional[str]): Boolean expression for pre-filtering (For example, "category == 'shoes'").
            output_fields (Optional[List[str]]): List of fields to include in results。If it is None，Usually only return ID and distance。
            partition_names (Optional[List[str]]): List of partitions to search。If it is None，then search the entire collection。
            timeout (Optional[float]): Operation timeout。
            **kwargs: Passed to collection.search other parameters (For example consistency_level)。
        Returns:
            Optional[List[SearchResult]]: List containing each query result，Return if failed None。
                                        Each SearchResult contains multiple Hit object。
        """
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Unable to get collection '{collection_name}' to perform search。")
            return None

        # Ensure collection is loaded
        if not self.load_collection(
            collection_name, timeout=timeout
        ):  # Attempt to load，Exit if failed
            logger.error(f"Load collection before search '{collection_name}' Failed。")
            return None

        logger.info(
            f"In collection '{collection_name}' Search in {len(query_vectors)} vectors (Field: {vector_field}, top_k: {limit})..."
        )
        try:
            # ensure output_fields Contains primary key field，so that subsequent retrieval is possible ID
            pk_field_name = collection.schema.primary_field.name
            if output_fields and pk_field_name not in output_fields:
                output_fields_with_pk = output_fields + [pk_field_name]
            elif not output_fields:
                # If output_fields for None, Milvus Default will return ID and distance
                output_fields_with_pk = None  # Let Milvus handle default
            else:
                output_fields_with_pk = output_fields

            search_result = collection.search(
                data=query_vectors,
                anns_field=vector_field,
                param=search_params,
                limit=limit,
                expr=expression,
                output_fields=output_fields_with_pk,  # Use list may includePK
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
            # search_result is List[SearchResult]
            # EachSearchResultcorresponds to aquery_vector
            # Each search result contains a hit list
            num_results = len(search_result) if search_result else 0
            logger.info(f"Search completed。Return {num_results} Group results。")

            # # Example：Record the hit count of the first query
            # if search_result and len(search_result[0]) > 0:
            #     logger.debug(f"First query vector hit {len(search_result[0])} results。")
            # logger.debug(f"First result example - ID: {search_result[0][0].id}, Distance: {search_result[0][0].distance}")

            return search_result  # Return original SearchResult list
        except MilvusException as e:
            logger.error(f"In collection '{collection_name}' Search failed in: {e}")
            return None
        except Exception as e:
            logger.error(f"In collection '{collection_name}' Unexpected error occurred during search in: {e}")
            return None

    def query(
        self,
        collection_name: str,
        expression: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Optional[List[Dict]]:
        """
        Query entities based on scalar field filter conditions。
        Args:
            collection_name (str): Target collection name。
            expression (str): Filter condition expression (e.g., "book_id in [100, 200]" or "word_count > 1000").
            output_fields (Optional[List[str]]): List of fields to return。If it is None，Usually return all scalar fields and primary key。
                                            Can include '*' to get all fields（including vectors，may be large）。
            partition_names (Optional[List[str]]): Query within specified partition。
            limit (Optional[int]): Maximum number of entities to return。Milvus to query of limit has a limit (e.g., 16384)，Need to note。
            offset (Optional[int]): Offset of returned results（for pagination）。
            timeout (Optional[float]): Operation timeout。
             **kwargs: Passed to collection.query other parameters (For example consistency_level)。
        Returns:
            Optional[List[Dict]]: List of entities meeting conditions (Each entity is a dictionary)，Return if failed None。
        """
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Unable to get collection '{collection_name}' to perform query。")
            return None

        # Query No need to pre-load collection into memory，but requires connection
        self._ensure_connected()

        # Milvus to query of limit Has internal limits，If the passed limit is too large，may need to batch query or adjust
        # Default may be 16384，Check pymilvus document or Milvus configuration
        effective_limit = limit
        # if limit and limit > 16384:
        #     logger.warning(f"query limit {limit} may exceed Milvus internal limits (usually is 16384)，Results may be truncated。")
        #     # effective_limit = 16384 # or handle pagination as needed

        logger.info(
            f"In collection '{collection_name}' Execute query in: '{expression}' (Limit: {effective_limit}, Offset: {offset})..."
        )
        try:
            # ensure output_fields Contains primary key，because query results may not contain by default（and search different）
            pk_field_name = collection.schema.primary_field.name
            if (
                output_fields
                and pk_field_name not in output_fields
                and "*" not in output_fields
            ):
                query_output_fields = output_fields + [pk_field_name]
            elif not output_fields:
                # If None, Attempt to get all non-vector fields + PK
                query_output_fields = [
                    f.name
                    for f in collection.schema.fields
                    if f.dtype != DataType.FLOAT_VECTOR
                    and f.dtype != DataType.BINARY_VECTOR
                ]
                if pk_field_name not in query_output_fields:
                    query_output_fields.append(pk_field_name)
            else:  # Already contains PK or '*'
                query_output_fields = output_fields

            query_results = collection.query(
                expr=expression,
                output_fields=query_output_fields,
                partition_names=partition_names,
                limit=effective_limit,  # Use potentially adjusted limit
                offset=offset,
                timeout=timeout,
                **kwargs,
            )
            # query_results is List[Dict]
            logger.info(f"Query completed。Return {len(query_results)} entities。")
            return query_results
        except MilvusException as e:
            logger.error(f"In collection '{collection_name}' Failed to execute query in: {e}")
            return None
        except Exception as e:
            logger.error(f"In collection '{collection_name}' Unexpected error occurred during query execution in: {e}")
            return None

    # --- Context Manager Support ---
    def __enter__(self):
        """support with Statement，Ensure connection upon entry。"""
        try:
            self._ensure_connected()  # Ensure connection，Will throw exception if failed
        except Exception as e:
            logger.error(f"Enter MilvusManager Connection failed when entering context manager: {e}")
            raise  # Rethrow exception，Prevent entry with block
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """support with Statement，Disconnect upon exit。"""
        try:
            self.disconnect()
        except Exception as e:
            logger.error(f"Exit MilvusManager Failed to disconnect when exiting context manager: {e}")
        # Can be based on exc_type parameters to decide whether to log exception information
        if exc_type:
            logger.error(
                f"MilvusManager Exception caught when exiting context manager: {exc_type.__name__}: {exc_val}"
            )
        # Return False Indicates if an exception occurs，Do not suppress exception propagation
