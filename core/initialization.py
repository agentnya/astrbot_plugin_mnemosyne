# -*- coding: utf-8 -*-
"""
Mnemosyne Plugin initialization logic
Includes configuration loading、Schema Definition、Milvus Connection and setup、Other component initialization, etc.。
"""

from typing import TYPE_CHECKING
import asyncio
import re
from pymilvus import CollectionSchema, FieldSchema, DataType

from astrbot.core.log import LogManager

# Import necessary types and modules
from .constants import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_DIM,
    PRIMARY_FIELD_NAME,
    VECTOR_FIELD_NAME,
    DEFAULT_OUTPUT_FIELDS,
)
from .tools import parse_address

from ..memory_manager.message_counter import MessageCounter
from ..memory_manager.vector_db.milvus_manager import MilvusManager
from ..memory_manager.embedding import OpenAIEmbeddingAPI, GeminiEmbeddingAPI
from ..memory_manager.context_manager import ConversationContextManager

# Type hints，Avoid circular imports
if TYPE_CHECKING:
    from ..main import Mnemosyne

# Get a logger dedicated to initialization
init_logger = LogManager.GetLogger(log_name="MnemosyneInit")


def initialize_config_check(plugin: "Mnemosyne"):
    """
    Some necessary parameter checks can be placed here
    """
    astrbot_config = plugin.context.get_config()
    # ------ Checknum_pairs ------
    num_pairs = plugin.config["num_pairs"]
    # num_pairsNeeds to be less than['provider_settings']['max_context_length']The number of configurations，If this configuration is-1，Then there is no limit。
    astrbot_max_context_length = astrbot_config["provider_settings"][
        "max_context_length"
    ]
    if (
        astrbot_max_context_length > 0 and num_pairs > 2 * astrbot_max_context_length
    ):  # Multiply here2Because the message count calculation rules are different
        raise ValueError(
            f"\nnum_pairsCannot be greater thanastrbotConfiguration of(Maximum number of conversations carried(Entries))\
                            The number of configurations:{2 * astrbot_max_context_length}，Otherwise, it may cause message loss。\n"
        )
    elif astrbot_max_context_length == 0:
        raise ValueError(
            f"\nastrbotConfiguration of(Maximum number of conversations carried(Entries))\
                            The number of configurations:{astrbot_max_context_length}Must be greater than0，OtherwiseMnemosyneThe plugin will not function properly\n"
        )
    # ------ num_pairs ------

    # ------ Checkcontexts_memory_len ------
    contexts_memory_len = plugin.config.get("contexts_memory_len", 0)
    if (
        astrbot_max_context_length > 0
        and contexts_memory_len > astrbot_max_context_length
    ):
        raise ValueError(
            f"\ncontexts_memory_lenCannot be greater thanastrbotConfiguration of(Maximum number of conversations carried(Entries))\
                            The number of configurations:{astrbot_max_context_length}。\n"
        )
    # ------ contexts_memory_len ------

def initialize_config_and_schema(plugin: "Mnemosyne"):
    """Parse configuration、Validate and define schema/Index parameters。"""
    init_logger.debug("Start initializing configuration and Schema...")
    try:
        embedding_dim = plugin.config.get("embedding_dim", DEFAULT_EMBEDDING_DIM)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("configuration 'embedding_dim' Must be a positive integer。")

        fields = [
            FieldSchema(
                name=PRIMARY_FIELD_NAME,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Unique memory identifier",
            ),
            FieldSchema(
                name="personality_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Role associated with memoryID",
            ),
            FieldSchema(
                name="session_id",
                dtype=DataType.VARCHAR,
                max_length=72,
                description="SessionID",
            ),  # Added length limit
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=4096,
                description="Memory content（Summary or snippet）",
            ),  # Added length limit
            FieldSchema(
                name=VECTOR_FIELD_NAME,
                dtype=DataType.FLOAT_VECTOR,
                dim=embedding_dim,
                description="Memory embedding vector",
            ),
            FieldSchema(
                name="create_time",
                dtype=DataType.INT64,
                description="Timestamp when memory is created（Unix epoch）",
            ),
        ]

        plugin.collection_name = plugin.config.get(
            "collection_name", DEFAULT_COLLECTION_NAME
        )
        plugin.collection_schema = CollectionSchema(
            fields=fields,
            description=f"Long-term memory storage: {plugin.collection_name}",
            primary_field=PRIMARY_FIELD_NAME,
            enable_dynamic_field=plugin.config.get(
                "enable_dynamic_field", False
            ),  # Whether dynamic fields are allowed
        )

        # Define index parameters
        plugin.index_params = plugin.config.get(
            "index_params",
            {
                "metric_type": "L2",  # Default metric type
                "index_type": "AUTOINDEX",  # Default index type
                "params": {},
            },
        )
        # Define search parameters
        plugin.search_params = plugin.config.get(
            "search_params",
            {
                "metric_type": plugin.index_params.get(
                    "metric_type", "L2"
                ),  # Must match index metric type
                "params": {"nprobe": 10},  # IVF_* Example search parameters of, AutoIndex Usually not needed
            },
        )

        plugin.output_fields_for_query = plugin.config.get(
            "output_fields", DEFAULT_OUTPUT_FIELDS
        )
        # Ensure the primary key is always in the output fields (Milvus May be included by default，But explicitly specifying is safer)
        # if PRIMARY_FIELD_NAME not in plugin.output_fields_for_query:
        #     plugin.output_fields_for_query.append(PRIMARY_FIELD_NAME)

        init_logger.debug(f"Collection Schema Definition completed: '{plugin.collection_name}'")
        init_logger.debug(f"Index parameters: {plugin.index_params}")
        init_logger.debug(f"Search parameters: {plugin.search_params}")
        init_logger.debug(f"Query output fields: {plugin.output_fields_for_query}")
        init_logger.debug("Configuration and Schema Initialization successful。")

    except Exception as e:
        init_logger.error(f"initialize configuration and Schema Failed: {e}", exc_info=True)
        raise  # Rethrow exception，So that in the main __init__ Capture in


def initialize_milvus(plugin: "Mnemosyne"):
    """
    initialize MilvusManager。
    Decide to connect to based on configuration Milvus Lite Or standard Milvus server，
    And perform necessary collection and index settings。
    """
    init_logger.debug("start initialization Milvus Connection and setup...")
    connect_args = {} # Used to collect and pass to MilvusManager parameters
    is_lite_mode = False # Mark whether it is Lite mode

    try:
        # 1. Priority check Milvus Lite configuration
        lite_path = plugin.config.get("milvus_lite_path","")

        # 2. Get standard Milvus Address configuration of
        milvus_address = plugin.config.get("address")

        if lite_path:
            # --- Detected Milvus Lite configuration ---
            init_logger.info(f"Detected Milvus Lite configuration，Will use local path: '{lite_path}'")
            connect_args["lite_path"] = lite_path
            is_lite_mode = True
            if milvus_address:
                init_logger.warning(f"Configured simultaneously 'milvus_lite_path' and 'address'，Will preferentially use Lite path，Ignore 'address' ('{milvus_address}')。")

        elif milvus_address:
            # --- Not configured Lite path，Use standard Milvus Address ---
            init_logger.info(f"Not configured Milvus Lite path，Will be based on 'address' Configuration connection standard Milvus: '{milvus_address}'")
            is_lite_mode = False
            # Determine address Is URI Or host:port
            if milvus_address.startswith(("http://", "https://", "unix:")):
                init_logger.debug(f"Address '{milvus_address}' Recognized as URI。")
                connect_args["uri"] = milvus_address
            else:
                init_logger.debug(f"Address '{milvus_address}' Will be parsed as host:port。")
                try:
                    host, port = parse_address(milvus_address) # Parse using utility functions
                    connect_args["host"] = host
                    connect_args["port"] = port
                except ValueError as e:
                    raise ValueError(
                        f"Parse standard Milvus Address '{milvus_address}' (host:port Format) Failed: {e}"
                    ) from e
        else:
            # --- Neither Lite Path nor standard address ---
            init_logger.warning(
                "Not configured Milvus Lite Path and standard Milvus Address。Please check configuration。Will use default configuration/AstrBot/data/mnemosyne_data"
            )

        # 3. Add general parameters (to Lite and Standard Both may be effective)
        #    Add database name (db_name)
        db_name = plugin.config.get("db_name", "default") # Provide default value 'default'
        # Only when db_name Is not 'default' Only explicitly added to parameters when，To keep it concise
        if db_name != "default":
            connect_args["db_name"] = db_name
            init_logger.info(f"Will attempt to connect to the database: '{db_name}'。")
        else:
            init_logger.debug("Will use the default database 'default'。")

        #    Set connection alias
        #    If not configured，Generate a default alias based on the collection name
        alias = plugin.config.get(
            "connection_alias", f"mnemosyne_{plugin.collection_name}"
        )
        connect_args["alias"] = alias
        init_logger.debug(f"Set Milvus Connection alias as: '{alias}'。")

        # 4. Add only applicable to standard Milvus parameters (If not Lite mode)
        if not is_lite_mode:
            init_logger.debug("For standard Milvus Add authentication and security settings to the connection（If configured）。")
            # Securely obtain authentication configuration dictionary，If not present, it is an empty dictionary
            auth_config = plugin.config.get("authentication", {})

            # Add optional authentication and security parameters
            added_auth_params = []
            for key in ["user", "password", "token", "secure"]:
                if key in auth_config and auth_config[key] is not None:
                    # Special handling 'secure'，Ensure it is a boolean value
                    if key == 'secure':
                        value = auth_config[key]
                        if isinstance(value, str):
                            # From string 'true'/'false' (Case insensitive) Convert to boolean
                            secure_bool = value.lower() == 'true'
                        else:
                            # Attempt to directly convert to boolean
                            secure_bool = bool(value)
                        connect_args[key] = secure_bool
                        added_auth_params.append(f"{key}={secure_bool}")
                    else:
                        connect_args[key] = auth_config[key]
                        # Do not log password and token Value of
                        if key not in ['password', 'token']:
                            added_auth_params.append(f"{key}={auth_config[key]}")
                        else:
                            added_auth_params.append(f"{key}=******") # Hide sensitive values

            if added_auth_params:
                init_logger.info(f"Added standard connection parameters from configuration: {', '.join(added_auth_params)}")
            else:
                init_logger.debug("No additional authentication or security configuration found。")

        else: # is_lite_mode is True
            # Check and warn：If in Lite Mode configured with inapplicable parameters
            auth_config = plugin.config.get("authentication", {})
            ignored_keys = [k for k in ["user", "password", "token", "secure"] if k in auth_config and auth_config[k] is not None]
            if ignored_keys:
                init_logger.warning(f"Currently is Milvus Lite mode，The following authentication in configuration/Security parameters will be ignored: {ignored_keys}")


        # 5. initialize MilvusManager
        loggable_connect_args = {
            k: v for k, v in connect_args.items() if k not in ['password', 'token']
        }
        init_logger.info(f"Prepare to initialize with the following parameters MilvusManager: {loggable_connect_args}")

        # create MilvusManager instance
        plugin.milvus_manager = MilvusManager(**connect_args)

        # 6. Check connection status
        if not plugin.milvus_manager or not plugin.milvus_manager.is_connected():
            mode_name = "Milvus Lite" if is_lite_mode else "standard Milvus"
            raise ConnectionError(
                f"initialize MilvusManager Or connect to {mode_name} Failed。Please check configuration and Milvus Instance status。"
            )

        mode_name = "Milvus Lite" if plugin.milvus_manager._is_lite else "standard Milvus"
        init_logger.info(f"Successfully connected to {mode_name} (alias: {alias})。")

        # 7. Set collection and index
        init_logger.debug("Start setting Milvus Collection and index...")
        setup_milvus_collection_and_index(plugin)
        init_logger.info("Milvus Collection and index setup process has been called。")

        init_logger.debug("Milvus Initialization process successfully completed。")

    except Exception as e:
        init_logger.error(f"Milvus Error occurred during initialization or setup: {e}", exc_info=True) # exc_info=True Will log stack trace
        plugin.milvus_manager = None  # Ensure when initialization fails manager Is set to None
        raise # Rethrow exception，So that upper-level code can capture and handle


def setup_milvus_collection_and_index(plugin: "Mnemosyne"):
    """ensure Milvus Collection and index exist and are loaded。"""
    if not plugin.milvus_manager or not plugin.collection_schema:
        init_logger.error("Cannot set Milvus Collection/Index：Manager or Schema not initialized。")
        raise RuntimeError("MilvusManager or CollectionSchema Not ready。")

    collection_name = plugin.collection_name

    # Check if collection exists
    if plugin.milvus_manager.has_collection(collection_name):
        init_logger.info(f"Collection '{collection_name}' Already exists。Start checking Schema Consistency...")
        check_schema_consistency(plugin, collection_name, plugin.collection_schema)
        # Note: check_schema_consistency Currently only log warnings，Do not block subsequent operations
    else:
        # If the collection does not exist，Then create the collection
        init_logger.info(f"Collection not found '{collection_name}'。Creating...")
        if not plugin.milvus_manager.create_collection(
            collection_name, plugin.collection_schema
        ):
            raise RuntimeError(f"create Milvus Collection '{collection_name}' Failed。")
        init_logger.info(f"Successfully created collection '{collection_name}'。")
        # Immediately attempt to create index after creating collection
        ensure_milvus_index(plugin, collection_name)

    # Ensure index exists again（Check even if the collection already exists）
    ensure_milvus_index(plugin, collection_name)

    # Ensure collection is loaded into memory for search
    init_logger.info(f"Ensure collection '{collection_name}' Loaded into memory...")
    if not plugin.milvus_manager.load_collection(collection_name):
        # Loading failure may be due to resource issues or index not ready，Print error but may allow plugin to continue（Depends on fault tolerance strategy）
        init_logger.error(
            f"Load collection '{collection_name}' Failed。Search functionality may not work properly or be inefficient。"
        )
        # Consider throwing an exception here，If loading is mandatory
        # raise RuntimeError(f"Load Milvus Collection '{collection_name}' Failed。")
    else:
        init_logger.info(f"Collection '{collection_name}' Successfully loaded。")


def ensure_milvus_index(plugin: "Mnemosyne", collection_name: str):
    """Check if the index for the vector field exists，If it does not exist, create it。"""
    if not plugin.milvus_manager:
        return

    try:
        has_vector_index = False
        # First check if the collection exists，Avoid errors in subsequent operations
        if not plugin.milvus_manager.has_collection(collection_name):
            init_logger.warning(
                f"Attempt for a non-existent collection '{collection_name}' Check/Create index，Skip。"
            )
            return

        # Check if the vector field has an index
        collection = plugin.milvus_manager.get_collection(collection_name)
        if collection:
            for index in collection.indexes:
                # Check if the index is created for our configured vector field
                if index.field_name == VECTOR_FIELD_NAME:
                    # optional：More strictly check if index type and parameters match configuration
                    # index_info = index.to_dict() if hasattr(index, 'to_dict') else {}
                    # configured_index_type = plugin.index_params.get('index_type')
                    # actual_index_type = index_info.get('index_type', index_info.get('index_param', {}).get('index_type')) # Compatible with different versions/API
                    # if configured_index_type and actual_index_type and configured_index_type != actual_index_type:
                    #     init_logger.warning(f"Collection '{collection_name}' Field '{VECTOR_FIELD_NAME}' Index type of ({actual_index_type}) With configuration ({configured_index_type}) Does not match。")
                    # else:
                    init_logger.info(
                        f"In collection '{collection_name}' Field detected on '{VECTOR_FIELD_NAME}' Existing index of。"
                    )
                    has_vector_index = True
                    break  # Exit loop once found
        else:
            init_logger.warning(
                f"Unable to get collection '{collection_name}' Object to verify index information in detail。"
            )

        # If no vector index is found，Then attempt to create
        if not has_vector_index:
            init_logger.warning(
                f"Collection '{collection_name}' Vector field of '{VECTOR_FIELD_NAME}' Index not yet created。Attempting to create..."
            )
            # Create index using configured index parameters
            index_success = plugin.milvus_manager.create_index(
                collection_name=collection_name,
                field_name=VECTOR_FIELD_NAME,
                index_params=plugin.index_params,
                # index_name=f"{VECTOR_FIELD_NAME}_idx" # Can specify index name，optional
                timeout=plugin.config.get(
                    "create_index_timeout", 600
                ),  # Increase timeout setting for index creation
            )
            if not index_success:
                init_logger.error(
                    f"For field '{VECTOR_FIELD_NAME}' Failed to create index。Search performance will be severely affected。please check Milvus Log。"
                )
                # As needed，Consider throwing an exception
            else:
                init_logger.info(
                    f"For field '{VECTOR_FIELD_NAME}' Sent index creation request。Index will be built in the background。"
                )
                # After creating the index，May need to wait for it to complete building for optimal performance，But can usually continue running
                # Consider adding a step to check index status，Or enforce before the first search load

    except Exception as e:
        init_logger.error(f"Check or create collection '{collection_name}' Error occurred when indexing: {e}")
        # Decide whether to rethrow the exception，This may prevent the plugin from starting
        raise


def initialize_components(plugin: "Mnemosyne"):
    """Initialize non Milvus Other components of，Such as context manager and embedding API。"""
    init_logger.debug("Start initializing other core components...")
    # 1. Initialize message counter and context manager
    try:
        plugin.context_manager = ConversationContextManager()
        plugin.msg_counter = MessageCounter()
        init_logger.info("Message counter initialized successfully。")
    except Exception as e:
        init_logger.error(f"Message counter initialization failed:{e}")
        raise

    # 2. initialize Embedding API
    try:
        # Check if necessary configuration exists
        try:
            plugin.ebd = plugin.context.get_registered_star("astrbot_plugin_embedding_adapter").star_cls
            dim=plugin.ebd.get_dim()
            modele_name=plugin.ebd.get_model_name()
            if dim is not None and modele_name is not None:
                plugin.config["embedding_dim"] = dim
                plugin.config["collection_name"] = "ea_"+re.sub(r'[^a-zA-Z0-9]', '_', modele_name)
        except Exception as e:
            init_logger.warning(f"embedding service adapter plugin load failed: {e}", exc_info=True)
            plugin.ebd = None

        required_keys = ["embedding_model", "embedding_key"]
        missing_keys = [key for key in required_keys if not plugin.config.get(key)]
        
        if plugin.ebd is None:
            if missing_keys:
                raise ValueError(f"Missing Embedding API configuration items: {', '.join(missing_keys)}")
            embedding_service = plugin.config.get("embedding_service")

            if embedding_service == "gemini":
                plugin.ebd = GeminiEmbeddingAPI(
                    model=plugin.config.get("embedding_model"),
                    api_key=plugin.config.get("embedding_key"),
                )
                init_logger.info("has been selected Gemini as embedding service provider")
            elif embedding_service == "openai":
                plugin.ebd = OpenAIEmbeddingAPI(
                    model=plugin.config.get("embedding_model"),
                    api_key=plugin.config.get("embedding_key"),
                    base_url=plugin.config.get("embedding_url"),
                )
                init_logger.info("has been selected OpenAI as embedding service provider")
            else:
                raise ValueError(f"Unsupported embedding service provider: {embedding_service}")

        try:
            plugin.ebd.test_connection()  # Assume this method throws an exception on failure
            init_logger.info("Embedding API Initialization successful，Connection test passed。")
        except AttributeError:
            init_logger.warning(
                "Embedding API Class does not have test_connection Method，Skip connection test。"
            )
        except Exception as conn_err:
            init_logger.error(f"Embedding API Connection test failed: {conn_err}", exc_info=True)
            # Decide whether to allow the plugin to Embedding API Continue running when connection fails
            # raise ConnectionError(f"Unable to connect to Embedding API: {conn_err}") from conn_err
            init_logger.warning("Will continue running，But Embedding Functionality will be unavailable。")
            plugin.ebd = None  # Explicitly set as None Indicates unavailable

    except Exception as e:
        init_logger.error(f"initialize Embedding API Failed: {e}", exc_info=True)
        plugin.ebd = None  # Ensure when failing ebd for None
        raise  # Embedding Is a core function，Failure means the plugin cannot work

    init_logger.debug("Other core components initialized。")


def check_schema_consistency(
    plugin: "Mnemosyne", collection_name: str, expected_schema: CollectionSchema
):
    """
    Check existing collection's Schema is consistent with expectations (Simplified version，Mainly check field names and types)。
    Log warning information，But do not prevent plugin from running。
    """
    if not plugin.milvus_manager or not plugin.milvus_manager.has_collection(
        collection_name
    ):
        # init_logger.info(f"Collection '{collection_name}' Does not exist，No need to check consistency。")
        return True  # No existing collection to compare

    try:
        collection = plugin.milvus_manager.get_collection(collection_name)
        if not collection:
            init_logger.error(f"Unable to get collection '{collection_name}' To check Schema。")
            return False  # Considered inconsistent

        actual_schema = collection.schema
        expected_fields = {f.name: f for f in expected_schema.fields}
        actual_fields = {f.name: f for f in actual_schema.fields}

        consistent = True
        warnings = []

        # Check if expected fields exist and basic types match
        for name, expected_field in expected_fields.items():
            if name not in actual_fields:
                warnings.append(f"Schema warning：Fields expected in configuration '{name}' Missing in actual collection")
                consistent = False
                continue  # Skip subsequent checks for this field

            actual_field = actual_fields[name]
            # Check data type
            if actual_field.dtype != expected_field.dtype:
                # Special handling for vector type，Check dimensions
                is_vector_expected = expected_field.dtype in [
                    DataType.FLOAT_VECTOR,
                    DataType.BINARY_VECTOR,
                ]
                is_vector_actual = actual_field.dtype in [
                    DataType.FLOAT_VECTOR,
                    DataType.BINARY_VECTOR,
                ]

                if is_vector_expected and is_vector_actual:
                    expected_dim = expected_field.params.get("dim")
                    actual_dim = actual_field.params.get("dim")
                    if expected_dim != actual_dim:
                        warnings.append(
                            f"Schema warning：Field '{name}' Vector dimensions do not match (Expected {expected_dim}, Actual {actual_dim})"
                        )
                        consistent = False
                elif (
                    expected_field.dtype == DataType.VARCHAR
                    and actual_field.dtype == DataType.VARCHAR
                ):
                    # Check VARCHAR of max_length
                    expected_len = expected_field.params.get("max_length")
                    actual_len = actual_field.params.get("max_length")
                    # If actual length is less than expected，May cause data truncation，Issue a warning
                    # If actual length is greater than expected，Usually not a problem，But may also prompt
                    if (
                        expected_len is not None
                        and actual_len is not None
                        and actual_len < expected_len
                    ):
                        warnings.append(
                            f"Schema warning：Field '{name}' of VARCHAR Insufficient length (Expected {expected_len}, Actual {actual_len})"
                        )
                        consistent = False  # This may be more serious
                    elif (
                        expected_len is not None
                        and actual_len is not None
                        and actual_len > expected_len
                    ):
                        warnings.append(
                            f"Schema hint：Field '{name}' of VARCHAR Length greater than expected (Expected {expected_len}, Actual {actual_len})"
                        )
                        # consistent = False # Usually not considered a serious issue
                else:
                    # Other type mismatches
                    warnings.append(
                        f"Schema warning：Field '{name}' Data type does not match (Expected {expected_field.dtype}, Actual {actual_field.dtype})"
                    )
                    consistent = False

            # Check primary key attributes
            if actual_field.is_primary != expected_field.is_primary:
                warnings.append(f"Schema warning：Field '{name}' Primary key status does not match")
                consistent = False
            # Check auto_id Attribute (Only meaningful when it is a primary key)
            if (
                expected_field.is_primary
                and actual_field.auto_id != expected_field.auto_id
            ):
                warnings.append(f"Schema warning：Primary key field '{name}' of AutoID Status does not match")
                consistent = False

        # Check if there are fields in the actual collection not defined in the configuration
        for name in actual_fields:
            if name not in expected_fields:
                # If dynamic fields are allowed，This may be normal
                if not plugin.collection_schema.enable_dynamic_field:
                    warnings.append(
                        f"Schema warning：Found fields not defined in configuration '{name}' (And dynamic fields not enabled)"
                    )
                    # consistent = False # Whether considered inconsistent depends on the policy

        if not consistent:
            warning_message = (
                f"Collection '{collection_name}' of Schema Potential inconsistency with current configuration:\n - "
                + "\n - ".join(warnings)
            )
            warning_message += "\nPlease check your Milvus Collection structure or plugin configuration。Inconsistency may cause runtime errors or data issues。"
            init_logger.warning(warning_message)
        else:
            init_logger.info(f"Collection '{collection_name}' of Schema Basically consistent with current configuration。")

        return consistent

    except Exception as e:
        init_logger.error(
            f"Check collection '{collection_name}' Schema Error occurred during consistency: {e}", exc_info=True
        )
        return False  # Treat error as inconsistency
