# -*- coding: utf-8 -*-
"""
Mnemosyne Implementation of the plugin's command handler function
(Note：Decorator has been removed，Function receives self)
"""

from typing import TYPE_CHECKING, Optional
from datetime import datetime

# import AstrBot API and type (Only requires event and message segment)
from astrbot.api.event import AstrMessageEvent

# Import necessary modules and constants
from .constants import PRIMARY_FIELD_NAME, MAX_TOTAL_FETCH_RECORDS

# Type hints
if TYPE_CHECKING:
    from ..main import Mnemosyne


async def list_collections_cmd_impl(self: "Mnemosyne", event: AstrMessageEvent):
    """[Implement] list current Milvus all collections in instance"""
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus Service not initialized or not connected。")
        return
    try:
        collections = self.milvus_manager.list_collections()
        if collections is None:
            yield event.plain_result("⚠️ Failed to get collection list，Please check the logs。")
            return
        if not collections:
            response = "Current Milvus No collections found in the instance。"
        else:
            response = "Current Milvus Collection list in the instance：\n" + "\n".join(
                [f"📚 {col}" for col in collections]
            )
            if self.collection_name in collections:
                response += f"\n\nCollection currently used by the plugin: {self.collection_name}"
            else:
                response += (
                    f"\n\n⚠️ Collection configured for the current plugin '{self.collection_name}' Not in the list！"
                )
        yield event.plain_result(response)
    except Exception as e:
        self.logger.error(f"Execute 'memory list' Command failed: {str(e)}", exc_info=True)
        yield event.plain_result(f"⚠️ Error occurred while getting collection list: {str(e)}")


async def delete_collection_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    collection_name: str,
    confirm: Optional[str] = None,
):
    """[Implement] delete specified Milvus collection and all its data"""
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus Service not initialized or not connected。")
        return

    is_current_collection = collection_name == self.collection_name
    warning_msg = ""
    if is_current_collection:
        warning_msg = f"\n\n🔥🔥🔥 Warning：You are attempting to delete a collection currently used by the plugin '{collection_name}'！This will cause plugin functionality to malfunction，Until recreated or configuration is changed！ 🔥🔥🔥"

    if confirm != "--confirm":
        yield event.plain_result(
            f"⚠️ Operation confirmation ⚠️\n"
            f"This operation will permanently delete Milvus Collection '{collection_name}' and all data it contains！This operation cannot be undone！\n"
            f"{warning_msg}\n\n"
            f"If you are sure to continue，Please execute the command again and add `--confirm` parameters:\n"
            f"`/memory drop_collection {collection_name} --confirm`"
        )
        return

    try:
        sender_id = event.get_sender_id()
        self.logger.warning(
            f"administrator {sender_id} Request to delete collection: {collection_name} (Confirm execution)"
        )
        if is_current_collection:
            self.logger.critical(
                f"administrator {sender_id} Deleting the collection currently used by the plugin '{collection_name}'！"
            )

        success = self.milvus_manager.drop_collection(collection_name)
        if success:
            msg = f"✅ Successfully deleted Milvus Collection '{collection_name}'。"
            if is_current_collection:
                msg += "\nCollection used by the plugin has been deleted，Please handle as soon as possible！"
            yield event.plain_result(msg)
            self.logger.warning(f"administrator {sender_id} Successfully deleted the collection: {collection_name}")
            if is_current_collection:
                self.logger.error(
                    f"Collection currently used by the plugin '{collection_name}' Has been deleted，Related functions will be unavailable。"
                )
        else:
            yield event.plain_result(
                f"⚠️ Delete collection '{collection_name}' Request has been sent，But Milvus Return failed。please check Milvus Log for detailed information。"
            )

    except Exception as e:
        self.logger.error(
            f"Execute 'memory drop_collection {collection_name}' Serious error occurred during command: {str(e)}",
            exc_info=True,
        )
        yield event.plain_result(f"⚠️ Serious error occurred while deleting collection: {str(e)}")


async def list_records_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    collection_name: Optional[str] = None,
    limit: int = 5,
):
    """[Implement] Query the latest memory records of the specified collection (Descending by creation time，Automatically get the latest)"""
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus Service not initialized or not connected。")
        return

    # Get the current session's session_id (If filtering by session is needed)
    session_id = await self.context.conversation_manager.get_curr_conversation_id(
        event.unified_msg_origin
    )
    # session_id = "session_1" # If testing a specific session or no session filtering，Can be hardcoded here or set as None

    target_collection = collection_name or self.collection_name

    # For user input limit Perform validation
    if limit <= 0 or limit > 50:
        # Limit the display number of user requests
        yield event.plain_result("⚠️ Display number (limit) Must be between 1 to 50 and。")
        return

    try:
        if not self.milvus_manager.has_collection(target_collection):
            yield event.plain_result(f"⚠️ Collection '{target_collection}' Does not exist。")
            return

        # Build query expression - Based only on session_id (If needed)
        if session_id:
            # If there is a sessionID，Then by sessionIDFilter
            expr = f'session_id in ["{session_id}"]'
            self.logger.info(
                f"Will be by session ID '{session_id}' Filter and query all related records (Limit {MAX_TOTAL_FETCH_RECORDS} Entries)。"
            )
        else:
            # If there is no sessionIDContext，Query all records
            expr = f"{PRIMARY_FIELD_NAME} >= 0"
            self.logger.info(
                "Session not specified ID，Will query the collection '{target_collection}' All records in (Limit {MAX_TOTAL_FETCH_RECORDS} Entries)。"
            )
            # Or，If your milvus_manager Supports empty expression to query all，Then expr = "" or None

        # self.logger.debug(f"Query collection '{target_collection}' Records: expr='{expr}'") # More specific logs are above
        output_fields = [
            "content",
            "create_time",
            "session_id",
            "personality_id",
            PRIMARY_FIELD_NAME,
        ]

        self.logger.debug(
            f"Prepare query Milvus: Collection='{target_collection}', Expression='{expr}', Limit={limit},Output fields={output_fields}, Total limit={MAX_TOTAL_FETCH_RECORDS}"
        )

        # Directly use Milvus of offset and limit Parameters for paginated query
        # records = self.milvus_manager.query(
        #     collection_name=target_collection,
        #     expression=expr,
        #     output_fields=output_fields,
        #     limit=limit,
        #     offset=offset,  # Directly use function parameters offset
        # )

        # Important modifications：Remove Milvus query of offset and limit parameters，Use total limit as Milvus of limit
        fetched_records = self.milvus_manager.query(
            collection_name=target_collection,
            expression=expr,
            output_fields=output_fields,
            limit=MAX_TOTAL_FETCH_RECORDS,  # Use total limit as Milvus of limit
        )

        # Check query results
        if fetched_records is None:
            # Query failed，milvus_manager.query Usually returns None or throw exception
            self.logger.error(
                f"Query collection '{target_collection}' Failed，milvus_manager.query Return None。"
            )
            yield event.plain_result(
                f"⚠️ Query collection '{target_collection}' Record failure，Please check the logs。"
            )
            return

        if not fetched_records:
            # Query successful，But no records were returned
            session_filter_msg = f"In session '{session_id}' in" if session_id else ""
            self.logger.info(
                f"Collection '{target_collection}' {session_filter_msg} No matching memory records found。"
            )
            yield event.plain_result(
                f"Collection '{target_collection}' {session_filter_msg} No matching memory records found in。"
            )
            return
        # Check if the total limit has been reached
        if len(fetched_records) >= MAX_TOTAL_FETCH_RECORDS:
            self.logger.warning(
                f"Number of records queried reached the total limit ({MAX_TOTAL_FETCH_RECORDS})，There may be more records not retrieved，Causing inability to find older records，But the latest records should be within the retrieval range。"
            )
            yield event.plain_result(
                f"ℹ️ Warning：Number of records queried has reached the system's limit for retrieving the latest records ({MAX_TOTAL_FETCH_RECORDS})。If there are many records，May not be able to display older content，But the latest records should be included。"
            )

        self.logger.debug(f"Successfully retrieved {len(fetched_records)} Original records for sorting。")
        # --- Sort after retrieving all results (Descending by creation time) ---
        # This ensures sorting is based on all retrieved records，Find the true latest records
        try:
            # Use lambda Expression by create_time Field sorting，If the field does not exist or is None，Default to 0
            fetched_records.sort(
                key=lambda x: x.get("create_time", 0) or 0, reverse=True
            )
            self.logger.debug(
                f"Has retrieved {len(fetched_records)} Records by create_time Descending order。"
            )
        except Exception as sort_e:
            self.logger.warning(
                f"Error occurred while sorting query results: {sort_e}。Display order may not be chronological。"
            )
            # If sorting fails，Continue processing，But not guaranteed to be chronological

        # --- Get the top limit Records after sorting ---
        # From the sorted fetched_records Take the top from limit Records after sorting
        display_records = fetched_records[:limit]

        # display_records Will not be empty，Unless fetched_records It is empty itself，
        # And fetched_records The empty case has been handled earlier。

        # Prepare response message
        total_fetched = len(fetched_records)
        display_count = len(display_records)
        # Message informs the user this is the latest record
        response_lines = [
            f"📜 Collection '{target_collection}' The latest memory record (Total retrieved {total_fetched} Records for sorting, Display the latest {display_count} Entries):"
        ]

        # Format each record for display
        # Use enumerate From 1 Start generating serial numbers
        for i, record in enumerate(display_records, start=1):
            ts = record.get("create_time")
            try:
                # According to Milvus Document，Query In the result time Is float Type of Unix timestamp（Seconds）。
                time_str = (
                    datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                    if ts is not None  # Check ts Exists and is not None
                    else "Unknown time"
                )
            except (TypeError, ValueError, OSError) as time_e:
                # Handle invalid or unparseable timestamps
                self.logger.warning(
                    f"Records {record.get(PRIMARY_FIELD_NAME, 'UnknownID')} Timestamp of '{ts}' Invalid or parsing error: {time_e}"
                )
                time_str = f"Invalid timestamp({ts})" if ts is not None else "Unknown time"

            content = record.get("content", "Content unavailable")
            # Truncate overly long content for optimal display
            content_preview = content[:200] + ("..." if len(content) > 200 else "")
            record_session_id = record.get("session_id", "Unknown session")
            persona_id = record.get("personality_id", "Unknown persona")
            pk = record.get(PRIMARY_FIELD_NAME, "UnknownID")  # Get primary key

            response_lines.append(
                f"#{i} [ID: {pk}]\n"  # Use from 1 Starting serial number
                f"  Time: {time_str}\n"
                f"  Persona: {persona_id}\n"
                f"  Session: {record_session_id}\n"
                f"  Content: {content_preview}"
            )

        # Send formatted results
        yield event.plain_result("\n\n".join(response_lines))

    except Exception as e:
        # Capture all other potential exceptions
        self.logger.error(
            f"Execute 'memory list_records' Unexpected error occurred during command (Collection: {target_collection}): {str(e)}",
            exc_info=True,  # Record the full error stack
        )
        yield event.plain_result("⚠️ Internal error occurred while querying memory records，Please contact the administrator。")


async def delete_session_memory_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    session_id: str,
    confirm: Optional[str] = None,
):
    """[Implement] delete specified session ID all related memory information"""
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus Service not initialized or not connected。")
        return

    if not session_id or not session_id.strip():
        yield event.plain_result("⚠️ Please provide the session to delete memory ID (session_id)。")
        return

    session_id_to_delete = session_id.strip().strip('"`')

    if confirm != "--confirm":
        yield event.plain_result(
            f"⚠️ Operation confirmation ⚠️\n"
            f"This operation will permanently delete the session ID '{session_id_to_delete}' In collection '{self.collection_name}' All memory information in！This operation cannot be undone！\n\n"
            f"To confirm deletion，Please execute the command again and add `--confirm` parameters:\n"
            f'`/memory delete_session_memory "{session_id_to_delete}" --confirm`'
        )
        return

    try:
        collection_name = self.collection_name
        expr = f'session_id == "{session_id_to_delete}"'
        sender_id = event.get_sender_id()
        self.logger.warning(
            f"administrator {sender_id} Request to delete session '{session_id_to_delete}' All memories of (Collection: {collection_name}, Expression: '{expr}') (Confirm execution)"
        )

        mutation_result = self.milvus_manager.delete(
            collection_name=collection_name, expression=expr
        )

        if mutation_result:
            delete_pk_count = (
                mutation_result.delete_count
                if hasattr(mutation_result, "delete_count")
                else "Unknown"
            )
            self.logger.info(
                f"Delete session request sent '{session_id_to_delete}' Memory request。Returned delete count（May be inaccurate）: {delete_pk_count}"
            )
            try:
                self.logger.info(
                    f"Refreshing (Flush) Collection '{collection_name}' To apply the delete operation..."
                )
                self.milvus_manager.flush([collection_name])
                self.logger.info(f"Collection '{collection_name}' Refresh complete。Delete operation has taken effect。")
                yield event.plain_result(
                    f"✅ Session successfully deleted ID '{session_id_to_delete}' All memory information of。"
                )
            except Exception as flush_err:
                self.logger.error(
                    f"Refresh collection '{collection_name}' Error occurred while applying deletion: {flush_err}",
                    exc_info=True,
                )
                yield event.plain_result(
                    f"⚠️ Delete request sent，But error occurred while refreshing collection to apply changes: {flush_err}。Deletion may not have fully taken effect。"
                )
        else:
            yield event.plain_result(
                f"⚠️ Delete session ID '{session_id_to_delete}' Memory request failed。please check Milvus Log。"
            )

    except Exception as e:
        self.logger.error(
            f"Execute 'memory delete_session_memory' Serious error occurred during command (Session ID: {session_id_to_delete}): {str(e)}",
            exc_info=True,
        )
        yield event.plain_result(f"⚠️ Serious error occurred while deleting session memory: {str(e)}")


async def get_session_id_cmd_impl(self: "Mnemosyne", event: AstrMessageEvent):
    """[Implement] get current session talking to you ID"""
    try:
        session_id = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        if session_id:
            yield event.plain_result(f"Current session ID: {session_id}")
        else:
            yield event.plain_result(
                "🤔 Unable to get current session ID。May not have started a conversation yet，Or the session has ended/Invalid。"
            )
            self.logger.warning(
                f"user {event.get_sender_id()} in {event.unified_msg_origin} Attempt to get session_id Failed。"
            )
    except Exception as e:
        self.logger.error(
            f"Execute 'memory get_session_id' Command failed: {str(e)}", exc_info=True
        )
        yield event.plain_result(f"⚠️ Get current session ID Error occurred: {str(e)}")
