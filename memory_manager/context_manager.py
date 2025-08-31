from typing import List, Dict, Optional
import time
from astrbot.core.log import LogManager
from astrbot.api.event import AstrMessageEvent

class ConversationContextManager:
    """
    Session Context Manager
    """

    def __init__(self):
        self.conversations: Dict[str, Dict] = {}

    def init_conv(self, session_id:str, contexts:list[Dict],event:AstrMessageEvent):
        """
        FromAstrBotRetrieve Historical Messages
        """
        if session_id in self.conversations:
            return
        self.conversations[session_id] = {}
        self.conversations[session_id]["history"] = contexts
        self.conversations[session_id]["event"] = event
        # Initialize the Time of the Last Summary，This Will Be Lost Upon Restart，But Ignore It for Now
        # Restart the Timer Upon Restart，Restart the Timer When the User Talks Again，emmmm，Change It Later，Add aTODO
        # TODO Consider Whether to Save to the Database，Or Save to a File
        self.conversations[session_id]["last_summary_time"] = time.time()
        return

    def add_message(self, session_id: str, role: str, content: str) -> Optional[str]:
        """
        Add Conversation Message
        :param session_id: SessionID
        :param role: Role（user/assistant）
        :param content: Conversation Content
        :return: Return the Content String to Be Summarized When the Threshold Is Reached，Otherwise Return None
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "history": [],
                "last_summary_time": time.time(),
            }

        conversation = self.conversations[session_id]
        conversation["history"].append(
            {
                "role": role,
                "content": content,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),# This Will Not Be Included in the Summary Content，Should
            }
        )

    def get_summary_time(self, session_id: str) -> float:
        """
        Get the Last Summary Time
        """
        if session_id in self.conversations:
            return self.conversations[session_id]["last_summary_time"]
        else:
            return 0

    def update_summary_time(self, session_id: str):
        """
        Update the Last Summary Time
        """
        if session_id in self.conversations:
            self.conversations[session_id]["last_summary_time"] = time.time()

    def get_history(self, session_id: str) -> List[Dict]:
        """
        Get Conversation History
        :param session_id: SessionID
        :return: Conversation History
        """
        if session_id in self.conversations:
            return self.conversations[session_id]["history"]
        else:
            return []

    def get_session_context(self, session_id: str):
        """
        Retrievesession_idAll Corresponding Information
        """
        if session_id in self.conversations:
            return self.conversations[session_id]
        else:
            return {}