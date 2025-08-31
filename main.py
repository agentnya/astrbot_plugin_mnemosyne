# -*- coding: utf-8 -*-
"""
Mnemosyne - based on RAG of AstrBot long-term memory plugin main file
responsible for plugin registration、initialization process call、event and command binding。
"""

import asyncio
from typing import List, Optional, Union
import re


# --- AstrBot core import ---
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.star import Context, Star, register
from astrbot.api.all import *  # import AstrBot API
from astrbot.api.message_components import *  # import message component
from astrbot.core.log import LogManager
from astrbot.api.provider import LLMResponse, ProviderRequest

# --- plugin internal module import ---
from .core import initialization  # import initialization logic module
from .core import memory_operations  # import memory operation logic module
from .core import commands  # import command processing implementation module
from .core.constants import *  # import all constants
from .core.tools import is_group_chat

# --- type definitions and dependency libraries ---
from pymilvus import CollectionSchema
from .memory_manager.message_counter import MessageCounter
from .memory_manager.vector_db.milvus_manager import MilvusManager
from .memory_manager.embedding import OpenAIEmbeddingAPI, GeminiEmbeddingAPI
from .memory_manager.context_manager import ConversationContextManager


@register(
    "Mnemosyne",
    "lxfight",
    "aAstrBotplugin，implement based onRAGtechnology's long-term memory function。",
    "0.5.1",
    "https://github.com/lxfight/astrbot_plugin_mnemosyne",
)
class Mnemosyne(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.context = context
        self.logger = LogManager.GetLogger(log_name="Mnemosyne")

        # --- initialize core component state ---
        self.collection_schema: Optional[CollectionSchema] = None
        self.index_params: dict = {}
        self.search_params: dict = {}
        self.output_fields_for_query: List[str] = []
        self.collection_name: str = DEFAULT_COLLECTION_NAME
        self.milvus_manager: Optional[MilvusManager] = None
        self.msg_counter: Optional[MessageCounter] = None
        self.context_manager: Optional[ConversationContextManager] = None
        self.ebd: Optional[Union[OpenAIEmbeddingAPI, GeminiEmbeddingAPI, Star]] = None
        self.provider = None

        # initialize embedding service
        try:
            embedding_plugin = self.context.get_registered_star(
                "astrbot_plugin_embedding_adapter"
            )
            if embedding_plugin:
                self.ebd = embedding_plugin.star_cls
                dim = self.ebd.get_dim()
                modele_name = self.ebd.get_model_name()
                if dim is not None and modele_name is not None:
                    self.config["embedding_dim"] = dim
                    self.config["collection_name"] = "ea_" + re.sub(
                        r"[^a-zA-Z0-9]", "_", modele_name
                    )
            else:
                raise ValueError(
                    "embedding service adapter not properly registered or did not return valid dimensions and model name。"
                )
        except Exception as e:
            self.logger.warning(f"embedding service adapter plugin load failed: {e}", exc_info=True)
            self.ebd = None

        if self.ebd is None:
            try:
                embedding_service = config.get("embedding_service", "openai").lower()
                if embedding_service == "gemini":
                    self.ebd = GeminiEmbeddingAPI(
                        model=config.get("embedding_model", "gemini-embedding-exp-03-07"),
                        api_key=config.get("embedding_key"),
                    )
                    self.logger.info("has been selected Gemini as embedding service provider")
                else:
                    self.ebd = OpenAIEmbeddingAPI(
                        model=config.get("embedding_model", "text-embedding-3-small"),
                        api_key=config.get("embedding_key"),
                        base_url=config.get("embedding_url"),
                    )
                    self.logger.info("has been selected OpenAI as embedding service provider")
            except Exception as e:
                self.logger.error(
                    f"failed to initialize embedding service: {e}. please check if the configuration or embedding service plugin is correctly installed。",
                    exc_info=True,
                )
                self.ebd = None

        # --- a damn timer ---
        self._summary_check_task: Optional[asyncio.Task] = None

        summary_check_config = config.get("summary_check_task")
        self.summary_check_interval: int = summary_check_config.get(
            "SUMMARY_CHECK_INTERVAL_SECONDS", DEFAULT_SUMMARY_CHECK_INTERVAL_SECONDS
        )
        self.summary_time_threshold: int = summary_check_config.get(
            "SUMMARY_TIME_THRESHOLD_SECONDS", DEFAULT_SUMMARY_TIME_THRESHOLD_SECONDS
        )
        if self.summary_time_threshold <= 0:
            self.logger.warning(
                f"configured SUMMARY_TIME_THRESHOLD_SECONDS ({self.summary_time_threshold}) invalid，will disable time-based auto-summary。"
            )
            self.summary_time_threshold = float("inf")
        # need to refresh
        self.flush_after_insert = False

        self.logger.info("start initialization Mnemosyne plugin...")
        try:
            initialization.initialize_config_check(self)
            initialization.initialize_config_and_schema(self)  # initialize configuration andschema
            initialization.initialize_milvus(self)  # initialize Milvus
            initialization.initialize_components(self)  # initialize core components
            # --- start background summary check task ---
            if self.context_manager and self.summary_time_threshold != float("inf"):
                # ensure context_manager initialized and threshold valid
                self._summary_check_task = asyncio.create_task(
                    memory_operations._periodic_summarization_check(self)
                )
                self.logger.info("background summary check task started。")
            elif self.summary_time_threshold == float("inf"):
                self.logger.info("time-based auto-summary disabled，do not start background check task。")
            else:
                self.logger.warning(
                    "Context manager not initialized，unable to start background summary check task。"
                )

            self.logger.info("Mnemosyne plugin core components initialized successfully。")
        except Exception as e:
            self.logger.critical(
                f"Mnemosyne serious error occurred during plugin initialization，plugin may not work properly: {e}",
                exc_info=True,
            )

    # --- event handling hook (call memory_operations.py implementation in) ---
    @filter.on_llm_request()
    async def query_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        """[event hook] in LLM before request，query and inject long-term memory。"""
        # when session first occurs，plugin will fetch fromAstrBotcontext history，subsequent session history automatically managed by plugin
        try:
            if not self.provider:
                provider_id = self.config.get("LLM_providers", "")
                self.provider = self.context.get_provider_by_id(provider_id)

            await memory_operations.handle_query_memory(self, event, req)
        except Exception as e:
            self.logger.error(
                f"processing on_llm_request exception caught during hook: {e}", exc_info=True
            )
        return

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """[event hook] in LLM after response"""
        try:
            await memory_operations.handle_on_llm_resp(self, event, resp)
        except Exception as e:
            self.logger.error(
                f"processing on_llm_response exception caught during hook: {e}", exc_info=True
            )
        return

    # --- command processing (define method and apply decorator，call commands.py implementation in) ---

    @command_group("memory")
    def memory_group(self):
        """long-term memory management command group /memory"""
        # this method body is empty，mainly to define group
        pass

    # apply decorator，and call implementation function
    @memory_group.command("list")  # type: ignore
    async def list_collections_cmd(self, event: AstrMessageEvent):
        """list current Milvus all collections in instance /memory list
        usage example：/memory list
        """
        # call commands.py implementation in，and proxy yield
        async for result in commands.list_collections_cmd_impl(self, event):
            yield result
        return

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("drop_collection")  # type: ignore
    async def delete_collection_cmd(
        self,
        event: AstrMessageEvent,
        collection_name: str,
        confirm: Optional[str] = None,
    ):
        """[administrator] delete specified Milvus collection and all its data
        usage example：/memory drop_collection [collection_name] [confirm]
        """
        async for result in commands.delete_collection_cmd_impl(
            self, event, collection_name, confirm
        ):
            yield result

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("list_records")  # type: ignore
    async def list_records_cmd(
        self,
        event: AstrMessageEvent,
        collection_name: Optional[str] = None,
        limit: int = 5,
    ):
        """query memory records of specified collection (display in descending order of creation time)
        usage example: /memory list_records [collection_name] [limit]
        """
        async for result in commands.list_records_cmd_impl(
            self, event, collection_name, limit
        ):
            yield result
        return

    @permission_type(PermissionType.ADMIN)
    @memory_group.command("delete_session_memory")  # type: ignore
    async def delete_session_memory_cmd(
        self, event: AstrMessageEvent, session_id: str, confirm: Optional[str] = None
    ):
        """[administrator] delete specified session ID all related memory information
        usage example：/memory delete_session_memory [session_id] [confirm]
        """
        async for result in commands.delete_session_memory_cmd_impl(
            self, event, session_id, confirm
        ):
            yield result
        return

    @permission_type(PermissionType.MEMBER)
    @memory_group.command("reset")
    async def reset_session_memory_cmd(
        self, event: AstrMessageEvent, confirm: Optional[str] = None
    ):
        """clear current session ID memory information
        usage example：/memory reset [confirm]
        """
        if not self.context._config.get("platform_settings").get("unique_session"):
            if is_group_chat(event):
                yield event.plain_result("⚠️ group chat session isolation not enabled，prohibit clearing group chat long-term memory")
                return
        session_id = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        async for result in commands.delete_session_memory_cmd_impl(
            self, event, session_id, confirm
        ):
            yield result
        return

    @memory_group.command("get_session_id")  # type: ignore
    async def get_session_id_cmd(self, event: AstrMessageEvent):
        """get current session talking to you ID
        usage example：/memory get_session_id
        """
        async for result in commands.get_session_id_cmd_impl(self, event):
            yield result
        return

    # --- plugin lifecycle method ---
    async def terminate(self):
        """cleanup logic when plugin stops"""
        self.logger.info("Mnemosyne plugin is stopping...")
        # --- stop background summary check task ---
        if self._summary_check_task and not self._summary_check_task.done():
            self.logger.info("cancelling background summary check task...")
            self._summary_check_task.cancel()  # <--- cancel task
            try:
                # wait for task to actually cancel，set a timeout to avoid hanging
                await asyncio.wait_for(self._summary_check_task, timeout=5.0)
            except asyncio.CancelledError:
                self.logger.info("background summary check task successfully cancelled。")
            except asyncio.TimeoutError:
                self.logger.warning("wait for background summary check task cancel timeout。")
            except Exception as e:
                # catch other exceptions that may be thrown during task cancellation
                self.logger.error(f"error occurred while waiting for background task cancellation: {e}", exc_info=True)
        self._summary_check_task = None  # cleanup task references

        if self.milvus_manager and self.milvus_manager.is_connected():
            try:
                if (
                    not self.milvus_manager._is_lite
                    and self.milvus_manager.has_collection(self.collection_name)
                ):
                    self.logger.info(
                        f"releasing collection from memory '{self.collection_name}'..."
                    )
                    self.milvus_manager.release_collection(self.collection_name)

                self.logger.info("disconnecting from Milvus connection...")
                self.milvus_manager.disconnect()
                self.logger.info("Milvus connection successfully disconnected。")

            except Exception as e:
                self.logger.error(f"when stopping plugin with Milvus interaction error: {e}", exc_info=True)
        else:
            self.logger.info("Milvus manager not initialized or disconnected，no need to disconnect。")
        self.logger.info("Mnemosyne plugin has stopped。")
        return
