# -*- coding: utf-8 -*-
"""
Mnemosyne Core memory operation logic of the plugin
Include RAG query、LLM Response handling、Memory summarization and storage。
"""

import time
import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, List, Dict, Optional, Any

from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.event import AstrMessageEvent
from pymilvus.exceptions import MilvusException

from .tools import (
    remove_mnemosyne_tags,
    remove_system_mnemosyne_tags,
    remove_system_content,
    format_context_to_string,
)

# Import necessary types and modules
from .constants import (
    VECTOR_FIELD_NAME,
    DEFAULT_TOP_K,
    DEFAULT_MILVUS_TIMEOUT,
    DEFAULT_PERSONA_ON_NONE,
)

# Type hints，Avoid circular imports
if TYPE_CHECKING:
    from ..main import Mnemosyne

from astrbot.core.log import LogManager

logger = LogManager.GetLogger(__name__)


async def handle_query_memory(
    plugin: "Mnemosyne", event: AstrMessageEvent, req: ProviderRequest
):
    """
    processing LLM Before the request RAG Retrieval logic。
    Retrieve related long-term memory，And inject it into ProviderRequest in。
    """
    # logger = plugin.logger

    # --- Pre-check ---
    if not await _check_rag_prerequisites(plugin):
        return

    try:
        # --- Get session and persona information ---
        persona_id = await _get_persona_id(plugin, event)
        session_id = await plugin.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        # Determine if it is in the historical session manager，If not，Then initialize
        if session_id not in plugin.context_manager.conversations:
            plugin.context_manager.init_conv(session_id, req.contexts, event)

        # Clean memory tags
        clean_contexts(plugin, req)

        # Add user message
        plugin.context_manager.add_message(session_id, "user", req.prompt)
        # Counter+1
        plugin.msg_counter.increment_counter(session_id)

        # --- RAG Search ---
        detailed_results = []
        try:
            # 1. Vectorize user query
            try:
                query_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: plugin.ebd.get_embeddings([req.prompt]),

                )
            except Exception as e:
                logger.error(f"Execute Embedding Error occurred while fetching: {e}", exc_info=True)
                query_embeddings = None  # Ensure subsequent failure handling

            if not query_embeddings:
                logger.error("Unable to get user query's Embedding Vector。")
                return
            query_vector = query_embeddings[0]

            # 2. Execute Milvus Search
            detailed_results = await _perform_milvus_search(
                plugin, query_vector, session_id, persona_id
            )

            # 3. Format results and inject into prompt
            if detailed_results:
                _format_and_inject_memory(plugin, detailed_results, req)

        except Exception as e:
            logger.error(f"Handle long-term memory RAG Error occurred during query: {e}", exc_info=True)
            return

    except Exception as e:
        logger.error(f"processing LLM Memory query process before request failed: {e}", exc_info=True)


async def handle_on_llm_resp(
    plugin: "Mnemosyne", event: AstrMessageEvent, resp: LLMResponse
):
    """
    processing LLM Post-response logic。Update counter。
    """
    if resp.role != "assistant":
        logger.warning("LLM Response is not from assistant role，Do not record。")
        return

    try:
        session_id = await plugin.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        if not session_id:
            logger.error("Unable to get current session_id,Unable to record LLM Response toMnemosyne。")
            return
        persona_id = await _get_persona_id(plugin, event)

        # Determine if summarization is needed
        await _check_and_trigger_summary(
            plugin,
            session_id,
            plugin.context_manager.get_history(session_id),
            persona_id,
        )

        plugin.logger.debug(f"Returned content：{resp.completion_text}")
        plugin.context_manager.add_message(
            session_id, "assistant", resp.completion_text
        )
        plugin.msg_counter.increment_counter(session_id)

    except Exception as e:
        logger.error(f"processing LLM Post-response memory recording failed: {e}", exc_info=True)


# Memory query (RAG) Related functions
async def _check_rag_prerequisites(plugin: "Mnemosyne") -> bool:
    """
    Check RAG Whether the query preconditions are met。

    Args:
        plugin: Mnemosyne Plugin instance。

    Returns:
        True If preconditions are met，False Otherwise。
    """
    # logger = plugin.logger
    if not plugin.milvus_manager or not plugin.milvus_manager.is_connected():
        logger.error("Milvus Service not initialized or not connected，Unable to query long-term memory。")
        return False
    if not plugin.ebd:
        logger.error("Embedding API not initialized，Unable to query long-term memory。")
        return False
    if not plugin.msg_counter:
        logger.error("Message counter not initialized，Will not be able to achieve memory summarization")
        return False
    return True


async def _get_persona_id(
    plugin: "Mnemosyne", event: AstrMessageEvent
) -> Optional[str]:
    """
    Get the persona of the current session ID。

    Args:
        plugin: Mnemosyne Plugin instance。
        event: Message event。

    Returns:
        Persona ID String，If no persona or error occurs, then None。
    """
    # logger = plugin.logger
    session_id = await plugin.context.conversation_manager.get_curr_conversation_id(
        event.unified_msg_origin
    )
    conversation = await plugin.context.conversation_manager.get_conversation(
        event.unified_msg_origin, session_id
    )
    persona_id = conversation.persona_id if conversation else None

    if not persona_id or persona_id == "[%None]":
        default_persona = plugin.context.provider_manager.selected_default_persona
        persona_id = default_persona["name"] if default_persona else None
        if not persona_id:
            logger.warning(
                f"Current session (ID: {session_id}) And globally no persona configured，Will use placeholder '{DEFAULT_PERSONA_ON_NONE}' Perform memory operation（If persona filtering is enabled）。"
            )
            if plugin.config.get("use_personality_filtering", False):
                persona_id = DEFAULT_PERSONA_ON_NONE
            else:
                persona_id = None
        else:
            logger.info(f"Current session has no persona，Use default persona: '{persona_id}'")
    return persona_id


async def _check_and_trigger_summary(
    plugin: "Mnemosyne",
    session_id: str,
    context: List[Dict],
    persona_id: Optional[str],
):
    """
    Check if summarization conditions are met and trigger summarization task。

    Args:
        plugin: Mnemosyne Plugin instance。
        session_id: Session ID。
        context: Request context list。
        persona_id: Persona ID.
    """
    if plugin.msg_counter.adjust_counter_if_necessary(
        session_id, context
    ) and plugin.msg_counter.get_counter(session_id) >= plugin.config.get(
        "num_pairs", 10
    ):
        logger.info("Start summarizing historical dialogue...")
        history_contents = format_context_to_string(
            context, plugin.config.get("num_pairs", 10)
        )
        # logger.debug(f"Summarized part{history_contents}")

        asyncio.create_task(
            handle_summary_long_memory(plugin, persona_id, session_id, history_contents)
        )
        logger.info("Summarization of historical dialogue task submitted to background execution。")
        plugin.msg_counter.reset_counter(session_id)


async def _perform_milvus_search(
    plugin: "Mnemosyne",
    query_vector: List[float],
    session_id: Optional[str],
    persona_id: Optional[str],
) -> Optional[List[Dict]]:
    """
    Execute Milvus Vector search。

    Args:
        plugin: Mnemosyne Plugin instance。
        query_vector: Query vector。
        session_id: Session ID。
        persona_id: Persona ID。

    Returns:
        Milvus Search result list，If not found or error occurs, then None。
    """
    # logger = plugin.logger
    # Prevent potential errors from lack of filtering conditions
    filters = ["memory_id > 0"]
    if session_id:
        filters.append(f'session_id == "{session_id}"')
    else:
        logger.warning("Unable to get current session_id，Will not according to session Filter memory！")

    use_personality_filtering = plugin.config.get("use_personality_filtering", False)
    effective_persona_id_for_filter = persona_id
    if use_personality_filtering and effective_persona_id_for_filter:
        filters.append(f'personality_id == "{effective_persona_id_for_filter}"')
        logger.debug(f"Will use persona '{effective_persona_id_for_filter}' Filter memory。")
    elif use_personality_filtering:
        logger.debug("Persona filtering enabled，But currently no valid persona ID，Do not filter by persona。")

    search_expression = " and ".join(filters) if filters else ""
    collection_name = plugin.collection_name
    top_k = plugin.config.get("top_k", DEFAULT_TOP_K)
    timeout_seconds = plugin.config.get("milvus_search_timeout", DEFAULT_MILVUS_TIMEOUT)

    logger.info(
        f"Start in the collection '{collection_name}' Search for related memory (TopK: {top_k}, Filter: '{search_expression or 'None'}')"
    )

    try:
        search_results = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: plugin.milvus_manager.search(
                    collection_name=collection_name,
                    query_vectors=[query_vector],
                    vector_field=VECTOR_FIELD_NAME,
                    search_params=plugin.search_params,
                    limit=top_k,
                    expression=search_expression,
                    output_fields=plugin.output_fields_for_query,
                ),
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(f"Milvus Search timeout ({timeout_seconds} Seconds)，Operation canceled。")
        return None
    except MilvusException as me:
        logger.error(f"Milvus Search operation failed: {me}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Execute Milvus Unknown error occurred during search: {e}", exc_info=True)
        return None

    if not search_results or not search_results[0]:
        logger.info("Vector search did not find related memory。")
        return None
    else:
        # From search_results Get from Hits object
        hits = search_results[0]
        # Call new helper function to handle Hits Object and extract detailed results
        detailed_results = _process_milvus_hits(hits)
        return detailed_results


def _process_milvus_hits(hits) -> List[Dict[str, Any]]:
    """
    processing Milvus SearchResults In Hits object，Use index-based traversal
    Extract valid memory entity data。

    Args:
        hits: From Milvus Search results search_results[0] Obtained from Hits object。

    Returns:
        A list containing extracted memory entity dictionaries。If no valid entity is extracted，
        Then return an empty list []。
    """
    detailed_results: List[Dict[str, Any]] = []  # Initialize result list，Specify type

    # Use index traversal hits object，To bypass SequenceIterator Iteration issues
    if hits:  # ensure hits Object is not empty or None
        try:
            num_hits = len(hits)  # Get hit count
            logger.debug(f"Milvus Returned {num_hits} Original hit results。")

            # Use index for traversal
            for i in range(num_hits):
                try:
                    hit = hits[i]  # Get single by index Hit object

                    # Check hit Object and its entity Whether attributes exist and are valid
                    # Use hasattr More robust，Avoid in entity Error when attribute does not exist
                    if hit and hasattr(hit, "entity") and hit.entity:
                        # Extract entity data，Use .get() Avoid KeyError
                        # Assume entity.to_dict() In the returned dictionary there is "entity" Key
                        entity_data = hit.entity.to_dict().get("entity")
                        # If data is successfully extracted，Then add to result list
                        if entity_data:
                            detailed_results.append(entity_data)
                        else:
                            # If entity Data exists but extracted is empty，May be a data structure issue
                            logger.warning(
                                f"Hit result index {i} At entity Data is empty or invalid，Skipped。"
                            )
                    else:
                        # If hit or entity invalid，Then skip
                        logger.debug(f"Hit result index {i} Object at or entity invalid，Skipped。")

                except Exception as e:
                    # Handle access or handle single hit Errors that may occur
                    logger.error(
                        f"Handle index {i} Error occurred at hit result: {e}", exc_info=True
                    )
                    # Continue to next on error hit，Do not interrupt the entire process

        except Exception as e:
            # More serious errors when handling length or setting loop
            # If error occurs here，detailed_results May be incomplete or empty
            logger.error(f"Error occurred during index-based hit result processing: {e}", exc_info=True)

    # Record number of successfully processed and extracted memories
    logger.debug(f"Number of successfully processed and extracted memories: {len(detailed_results)} Entries。")

    return detailed_results


# LLM Response handling related functions
def _format_and_inject_memory(
    plugin: "Mnemosyne", detailed_results: List[Dict], req: ProviderRequest
):
    """
    Format search results and inject into ProviderRequest in。

    Args:
        plugin: Mnemosyne Plugin instance。
        detailed_results: Detailed search result list。
        req: ProviderRequest object。
    """
    # logger = plugin.logger
    if not detailed_results:
        logger.info("Did not find or obtain related long-term memory，Do not supplement。")
        return

    long_memory_prefix = plugin.config.get(
        "long_memory_prefix", "<Mnemosyne> Long-term memory fragments："
    )
    long_memory_suffix = plugin.config.get("long_memory_suffix", "</Mnemosyne>")
    long_memory = f"{long_memory_prefix}\n"

    for result in detailed_results:
        content = result.get("content", "Content missing")
        ts = result.get("create_time")
        try:
            time_str = (
                datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                if ts
                else "Unknown time"
            )
        except (TypeError, ValueError):
            time_str = f"timestamp: {ts}" if ts else "Unknown time"

        memory_entry_format = plugin.config.get(
            "memory_entry_format", "- [{time}] {content}"
        )
        long_memory += memory_entry_format.format(time=time_str, content=content) + "\n"

    long_memory += long_memory_suffix

    logger.info(f"Supplemented {len(detailed_results)} Long-term memory entries into prompt。")
    logger.debug(f"Supplement content:\n{long_memory}")

    injection_method = plugin.config.get("memory_injection_method", "user_prompt")

    # Clean inserted long-term memory content
    clean_contexts(plugin, req)
    if injection_method == "user_prompt":
        req.prompt = long_memory + "\n" + req.prompt

    elif injection_method == "system_prompt":
        req.system_prompt += long_memory

    elif injection_method == "insert_system_prompt":
        req.contexts.append({"role": "system", "content": long_memory})

    else:
        logger.warning(
            f"Unknown memory injection method '{injection_method}'，Will default append to user prompt。"
        )
        req.prompt = long_memory + "\n" + req.prompt


# Delete supplemented long-term memory function
def clean_contexts(plugin: "Mnemosyne", req: ProviderRequest):
    """
    Delete tags in long-term memory
    """
    injection_method = plugin.config.get("memory_injection_method", "user_prompt")
    contexts_memory_len = plugin.config.get("contexts_memory_len", 0)
    if injection_method == "user_prompt":
        req.contexts = remove_mnemosyne_tags(req.contexts, contexts_memory_len)
    elif injection_method == "system_prompt":
        req.system_prompt = remove_system_mnemosyne_tags(
            req.system_prompt, contexts_memory_len
        )
    elif injection_method == "insert_system_prompt":
        req.contexts = remove_system_content(req.contexts, contexts_memory_len)
    return


# Memory summarization related functions
async def _check_summary_prerequisites(plugin: "Mnemosyne", memory_text: str) -> bool:
    """
    Check if memory summarization preconditions are met。

    Args:
        plugin: Mnemosyne Plugin instance。
        memory_text: Memory text。

    Returns:
        True If preconditions are met，False Otherwise。
    """
    # logger = plugin.logger
    if not plugin.milvus_manager or not plugin.milvus_manager.is_connected():
        logger.error("Milvus Service unavailable，Unable to store summarized long-term memory。")
        return False
    if not plugin.ebd:
        logger.error("Embedding API Unavailable，Unable to vectorize summarized memory。")
        return False
    if not memory_text or not memory_text.strip():
        logger.warning("Attempt to summarize empty or blank memory text，Skip。")
        return False
    return True


async def _get_summary_llm_response(
    plugin: "Mnemosyne", memory_text: str
) -> Optional[LLMResponse]:
    """
    Request LLM Perform memory summarization。

    Args:
        plugin: Mnemosyne Plugin instance。
        memory_text: Memory text to be summarized。

    Returns:
        LLMResponse object，If request fails, then None。
    """
    # logger = plugin.logger
    llm_provider = plugin.provider
    # TODO This part of the logic is true history，Change it later
    try:
        if not llm_provider:
            # Ifplugin.providerIncorrect，At this time，Use currently usedLLMService provider，Avoid errors
            llm_provider = plugin.context.get_using_provider()
            if not llm_provider:
                logger.error("Unable to get for memory summarization LLM Provider。")
                return None
    except Exception as e:
        logger.error(f"Retrieve LLM Provider Error occurred: {e}", exc_info=True)
        return None

    long_memory_prompt = plugin.config.get(
        "long_memory_prompt",
        "Please summarize the following multi-turn dialogue history into a concise、Objective、Long-term memory entry containing key information:",
    )
    summary_llm_config = plugin.config.get("summary_llm_config", {})

    logger.debug(
        f"Request LLM Summarize short-term memory，tip: '{long_memory_prompt[:50]}...', Content length: {len(memory_text)}"
    )

    try:
        llm_response = await llm_provider.text_chat(
            prompt=memory_text,
            contexts=[{"role": "system", "content": long_memory_prompt}],
            **summary_llm_config,
        )
        logger.debug(f"LLM Summarize response raw data: {llm_response}")
        return llm_response
    except Exception as e:
        logger.error(f"LLM Summarization request failed: {e}", exc_info=True)
        return None


def _extract_summary_text(
    plugin: "Mnemosyne", llm_response: LLMResponse
) -> Optional[str]:
    """
    From LLM Extract and verify summary text from response。

    Args:
        plugin: Mnemosyne Plugin instance。
        llm_response: LLMResponse object。

    Returns:
        Summary text string，If extraction fails, then None。
    """
    # logger = plugin.logger
    completion_text = None
    if isinstance(llm_response, LLMResponse):
        completion_text = llm_response.completion_text
        # role = llm_response.role
    elif isinstance(llm_response, dict):
        completion_text = llm_response.get("completion_text")
        # role = llm_response.get("role")
    else:
        logger.error(f"LLM Summary returned unknown type of data: {type(llm_response)}")
        return None

    if not completion_text or not completion_text.strip():
        logger.error(f"LLM Summary response invalid or content empty。Original response: {llm_response}")
        return None

    summary_text = completion_text.strip()
    logger.info(f"LLM Successfully generated memory summary，Length: {len(summary_text)}")
    return summary_text


async def _store_summary_to_milvus(
    plugin: "Mnemosyne",
    persona_id: Optional[str],
    session_id: str,
    summary_text: str,
    embedding_vector: List[float],
):
    """
    Store summary text and vector into Milvus in。

    Args:
        plugin: Mnemosyne Plugin instance。
        persona_id: Persona ID。
        session_id: Session ID。
        summary_text: Summary text。
        embedding_vector: Of the summary text Embedding Vector。
    """
    # logger = plugin.logger
    collection_name = plugin.collection_name
    current_timestamp = int(time.time())

    effective_persona_id = (
        persona_id
        if persona_id
        else plugin.config.get("default_persona_id_on_none", DEFAULT_PERSONA_ON_NONE)
    )

    data_to_insert = [
        {
            "personality_id": effective_persona_id,
            "session_id": session_id,
            "content": summary_text,
            VECTOR_FIELD_NAME: embedding_vector,
            "create_time": current_timestamp,
        }
    ]

    logger.info(
        f"Prepare to collection '{collection_name}' Insert 1 Summary memory entries (Persona: {effective_persona_id}, Session: {session_id[:8]}...)"
    )
    # mutation_result = plugin.milvus_manager.insert(
    #     collection_name=collection_name,
    #     data=data_to_insert,
    # )
    # --- Modify insert call ---
    loop = asyncio.get_event_loop()
    mutation_result = None
    try:
        mutation_result = await loop.run_in_executor(
            None,  # Use default thread pool
            lambda: plugin.milvus_manager.insert(
                collection_name=collection_name, data=data_to_insert
            ),
        )
    except Exception as e:
        logger.error(f"To Milvus Error inserting summary memory: {e}", exc_info=True)

    if mutation_result and mutation_result.insert_count > 0:
        inserted_ids = mutation_result.primary_keys
        logger.info(f"Successfully inserted summary memory into Milvus。Insert ID: {inserted_ids}")

        try:
            logger.debug(
                f"Refreshing (Flush) Collection '{collection_name}' To ensure memory is immediately available..."
            )
            # plugin.milvus_manager.flush([collection_name])
            await loop.run_in_executor(
                None,  # Use default thread pool
                lambda: plugin.milvus_manager.flush([collection_name]),
            )
            logger.debug(f"Collection '{collection_name}' Refresh complete。")

        except Exception as flush_err:
            logger.error(
                f"Refresh collection '{collection_name}' Error occurred: {flush_err}",
                exc_info=True,
            )
    else:
        logger.error(
            f"Insert summary memory into Milvus Failed。MutationResult: {mutation_result}. LLM Reply: {summary_text[:100]}..."
        )


async def handle_summary_long_memory(
    plugin: "Mnemosyne", persona_id: Optional[str], session_id: str, memory_text: str
):
    """
    Use LLM Summarize short-term dialogue history into long-term memory，And vectorize it before storing Milvus。
    This is a background task。
    """
    # logger = plugin.logger

    # --- Pre-check ---
    if not await _check_summary_prerequisites(plugin, memory_text):
        return

    try:
        # 1. Request LLM Perform summarization
        llm_response = await _get_summary_llm_response(plugin, memory_text)
        if not llm_response:
            return

        # 2. Extract summary text
        summary_text = _extract_summary_text(plugin, llm_response)
        if not summary_text:
            return

        # 3. Get summary text's Embedding
        # embedding_vectors = plugin.ebd.get_embeddings(summary_text)
        try:
            embedding_vectors = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: plugin.ebd.get_embeddings([summary_text]),

            )
        except Exception as e:
            logger.error(
                f"Get summary text Embedding Error occurred: '{summary_text[:100]}...' - {e}",
                exc_info=True,
            )
            embedding_vectors = None  # Ensure subsequent failure handling

        if not embedding_vectors:
            logger.error(f"Unable to get summary text's Embedding: '{summary_text[:100]}...'")
            return
        embedding_vector = embedding_vectors[0]

        # 4. Store into Milvus
        await _store_summary_to_milvus(
            plugin, persona_id, session_id, summary_text, embedding_vector
        )
        return
    except Exception as e:
        logger.error(f"Serious error occurred during summarizing or storing long-term memory: {e}", exc_info=True)


# Timer
async def _periodic_summarization_check(plugin: "Mnemosyne"):
    """
    [background tasks] Periodically check and trigger timed-out session summarization
    """
    logger.info(
        f"Start periodic summarization check task，Check interval: {plugin.summary_check_interval}Seconds, Summarization time threshold: {plugin.summary_time_threshold}Seconds。"
    )
    while True:
        try:
            await asyncio.sleep(plugin.summary_check_interval)  # <--- Wait for specified interval

            if not plugin.context_manager or plugin.summary_time_threshold == float(
                "inf"
            ):
                # If context manager not initialized or threshold invalid，Then skip this check
                continue

            current_time = time.time()
            session_ids_to_check = list(plugin.context_manager.conversations.keys())

            # logger.debug(f"Start checking {len(session_ids_to_check)} Sessions' summarization timed out...")

            for session_id in session_ids_to_check:
                try:
                    session_context = plugin.context_manager.get_session_context(
                        session_id
                    )
                    if not session_context:  # Session may be removed during check
                        continue
                    if plugin.msg_counter.get_counter(session_id) <= 0:
                        logger.debug(f"Session {session_id} No new messages，Skip check。")
                        continue

                    last_summary_time = session_context["last_summary_time"]

                    if current_time - last_summary_time > plugin.summary_time_threshold:
                        # logger.debug(f"current_time {current_time} - last_summary_time {last_summary_time} : {current_time - last_summary_time}")
                        logger.info(
                            f"Session {session_id} Exceeded threshold since last summarization ({plugin.summary_time_threshold}Seconds)，Trigger forced summarization。"
                        )
                        # Run summarization
                        logger.info("Start summarizing historical dialogue...")
                        history_contents = format_context_to_string(
                            session_context["history"],
                            plugin.msg_counter.get_counter(session_id),
                        )
                        # logger.debug(f"Summarized part{history_contents}")
                        persona_id = await _get_persona_id(
                            plugin, session_context["event"]
                        )
                        asyncio.create_task(
                            handle_summary_long_memory(
                                plugin, persona_id, session_id, history_contents
                            )
                        )
                        logger.info("Summarization of historical dialogue task submitted to background execution。")

                        plugin.msg_counter.reset_counter(session_id)
                        plugin.context_manager.update_summary_time(session_id)

                except KeyError:
                    # Session in getting keys After、Deleted before processing，Is normal
                    logger.debug(f"Check session {session_id} When，Session has been removed。")
                except Exception as e:
                    logger.error(
                        f"Check or summarize session {session_id} Error occurred: {e}", exc_info=True
                    )

        except asyncio.CancelledError:
            logger.info("Periodic summarization check task canceled。")
            break  # Exit loop
        except Exception as e:
            # Capture unexpected errors in the loop itself，Prevent task from completely stopping
            logger.error(f"Error in main loop of periodic summarization check task: {e}", exc_info=True)
            # Can choose to wait a bit here，Avoid error spamming
            await asyncio.sleep(plugin.summary_check_interval)
