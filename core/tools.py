# -*- coding: utf-8 -*-
"""
Mnemosyne plugin utility function
"""

import functools
import re
from typing import List, Dict, Set, Union
from urllib.parse import urlparse

from astrbot.api.event import AstrMessageEvent


def parse_address(address: str):
    """
    parse address，extract hostname and port number。
    if the address has no protocol prefix，then add by default "http://"
    """
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address
    parsed = urlparse(address)
    host = parsed.hostname
    port = (
        parsed.port if parsed.port is not None else 19530
    )  # if no port is specified，Default use19530
    return host, port


def content_to_str(func):
    """
    implement a decorator，convert all input content to string and print
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        str_args = [str(arg) for arg in args]
        str_kwargs = {k: str(v) for k, v in kwargs.items()}
        print(
            f"Function '{func.__name__}' called with arguments: args={str_args}, kwargs={str_kwargs}"
        )
        return func(*str_args, **str_kwargs)

    return wrapper


def remove_mnemosyne_tags(
    contents: List[Dict[str, str]], contexts_memory_len: int = 0
) -> List[Dict[str, str]]:
    """
    remove using regular expressionLLMin the context<mnemosyne> </mnemosyne>tag pair。
    - contexts_memory_len > 0: retain the latestNtag pairs。
    - contexts_memory_len == 0: remove all tag pairs。
    - contexts_memory_len < 0: retain all tag pairs，make no deletions。
    """
    if contexts_memory_len < 0:
        return contents

    compiled_regex = re.compile(r"<Mnemosyne>.*?</Mnemosyne>", re.DOTALL)
    cleaned_contents: List[Dict[str, str]] = []

    if contexts_memory_len == 0:
        for content_item in contents:
            if isinstance(content_item, dict) and content_item.get("role") == "user":
                original_text = content_item.get("content", "")
                if not isinstance(original_text, str):
                    original_text = str(original_text)
                cleaned_text = compiled_regex.sub("", original_text)
                cleaned_contents.append({"role": "user", "content": cleaned_text})
            else:
                cleaned_contents.append(content_item)
    else:  # contexts_memory_len > 0
        all_mnemosyne_blocks: List[str] = []
        for content_item in contents:
            if isinstance(content_item, dict) and content_item.get("role") == "user":
                original_text = content_item.get("content", "")
                if isinstance(original_text, str):
                    found_blocks = compiled_regex.findall(original_text)
                    all_mnemosyne_blocks.extend(found_blocks)

        blocks_to_keep: Set[str] = set(all_mnemosyne_blocks[-contexts_memory_len:])

        def replace_logic(match: re.Match) -> str:
            block = match.group(0)
            return block if block in blocks_to_keep else ""

        for content_item in contents:
            if isinstance(content_item, dict) and content_item.get("role") == "user":
                original_text = content_item.get("content", "")
                
                # --- 【core modification】make independent if Change to elif，form mutually exclusive logic ---
                if isinstance(original_text, list):
                    # 1. if it is a list，add directly，then process the next entry
                    cleaned_contents.append({"role": "user", "content": original_text})
                elif compiled_regex.search(original_text):
                    # 2. Otherwise，if the content matches the regex，then clean and add
                    cleaned_text = compiled_regex.sub(replace_logic, original_text)
                    cleaned_contents.append({"role": "user", "content": cleaned_text})
                else:
                    # 3. Otherwise (neither a list，nor matches the regex)，add the original entry directly
                    cleaned_contents.append(content_item)
            else:
                cleaned_contents.append(content_item)

    return cleaned_contents


def remove_system_mnemosyne_tags(text: str, contexts_memory_len: int = 0) -> str:
    """
    remove using regular expressionLLMin the context system prompt<Mnemosyne> </Mnemosyne>tag pair。
    If contexts_memory_len > 0，then only retain the last contexts_memory_len tag pairs。
    """
    if not isinstance(text, str):
        return text  # if the input is not a string，return directly

    if contexts_memory_len < 0:
        return text

    compiled_regex = re.compile(r"<Mnemosyne>.*?</Mnemosyne>", re.DOTALL)

    if contexts_memory_len == 0:
        cleaned_text = compiled_regex.sub("", text)
    else:
        all_mnemosyne_blocks: List[str] = compiled_regex.findall(text)
        blocks_to_keep: Set[str] = set(all_mnemosyne_blocks[-contexts_memory_len:])

        def replace_logic(match: re.Match) -> str:
            block = match.group(0)
            return block if block in blocks_to_keep else ""

        if compiled_regex.search(text):
            cleaned_text = compiled_regex.sub(replace_logic, text)
        else:
            cleaned_text = text

    return cleaned_text


def remove_system_content(
    contents: List[Dict[str, str]], contexts_memory_len: int = 0
) -> List[Dict[str, str]]:
    """
    FromLLMremove older system prompts in the context ('role'='system' messages)，
    retain a specified number of the latest system messages，and maintain the overall message order。
    """
    if not isinstance(contents, list):
        return []
    if contexts_memory_len < 0:
        return contents

    system_message_indices = [
        i for i, msg in enumerate(contents)
        if isinstance(msg, dict) and msg.get("role") == "system"
    ]
    indices_to_remove: Set[int] = set()
    num_system_messages = len(system_message_indices)

    if num_system_messages > contexts_memory_len:
        num_to_remove = num_system_messages - contexts_memory_len
        indices_to_remove = set(system_message_indices[:num_to_remove])

    cleaned_contents = [
        msg for i, msg in enumerate(contents) if i not in indices_to_remove
    ]

    return cleaned_contents


def format_context_to_string(
    context_history: List[Union[Dict[str, str], str]], length: int = 10
) -> str:
    """
    extract the last from the context history 'length' user entries andAIdialogue messages，
    and convert their content into a newline-separated string。
    """
    if length <= 0:
        return ""

    selected_contents: List[str] = []
    count = 0

    for message in reversed(context_history):
        if count >= length:
            break
            
        role = None
        content = None

        if isinstance(message, dict) and "role" in message and "content" in message:
            role = message.get("role")
            content = message.get("content")

        if content is not None:
            if role == "user":
                selected_contents.insert(0, str("user:" + str(content) + "\n"))
                count += 1
            elif role == "assistant":
                selected_contents.insert(0, str("assistant:" + str(content) + "\n"))
                count += 1

    return "\n".join(selected_contents)


def is_group_chat(event: AstrMessageEvent) -> bool:
    """
    determine if the message is from a group chat。
    """
    return event.get_group_id() != ""
