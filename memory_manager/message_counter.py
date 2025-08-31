import sqlite3
import os

from astrbot.core.log import LogManager

logging = LogManager.GetLogger(log_name="Message Counter")


class MessageCounter:
    """
    Message Counter Class，Use SQLite Store the message round count for each session。
    """

    def __init__(self, db_file=None):
        """
        Initialize message counter，Use SQLite Database storage。
        db_file Parameters are now optional。If it is None，Then automatically generate the database file path。

        Args:
            db_file (str, optional): SQLite Database file path。
                                     If it is None，Then automatically generate the path。
        """
        if db_file is None:
            # Get the current file directory
            current_file_dir = os.path.dirname(os.path.abspath(__file__))

            base_dir = current_file_dir
            for _ in range(3):
                base_dir = os.path.dirname(base_dir)
                # Avoid going all the way up beyond the root directory，A check can be added，If it is already the root directory，Then stop
                if (
                    os.path.dirname(base_dir) == base_dir
                ):  # Check if the root directory has been reached (dirname(root) == root)
                    break  # Stop going up

            # Build mnemosyne_data Folder path
            data_dir = os.path.join(base_dir, "mnemosyne_data")

            # ensure mnemosyne_data Folder exists，Create if it does not exist
            os.makedirs(
                data_dir, exist_ok=True
            )  # exist_ok=True Indicates if the directory already exists，No exception will be thrown

            self.db_file = os.path.join(data_dir, "message_counters.db")
        else:
            self.db_file = db_file  # If the user explicitly provides db_file，Then use the user-provided path

        self._initialize_db()

    def _initialize_db(self):
        """
        initialize SQLite Database and table。
        If the table does not exist，Then create 'message_counts' Table。
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS message_counts (
                    session_id TEXT PRIMARY KEY,
                    count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()
            logging.debug(f"SQLite Database initialization complete，File path: {self.db_file}")
        except sqlite3.Error as e:
            logging.error(f"initialize SQLite Database failure: {e}")
            if conn:
                conn.rollback()  # Rollback transaction
        finally:
            if conn:
                conn.close()

    def reset_counter(self, session_id):
        """
        Reset specified session ID Message counter of。
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO message_counts (session_id, count) VALUES (?, ?)",
                (session_id, 0),
            )
            conn.commit()
            logging.debug(f"Session {session_id} Counter has been reset to 0。")
        except sqlite3.Error as e:
            logging.error(f"Reset session {session_id} Database error occurred when resetting counter: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def increment_counter(self, session_id):
        """
        For the specified session ID Add to the message counter of 1。

        Args:
            session_id (str): Session ID。
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO message_counts (session_id, count) VALUES (?, 0)",
                (session_id,),
            )  # Insert if it does not exist，Initial value is0
            cursor.execute(
                "UPDATE message_counts SET count = count + 1 WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            logging.debug(f"Session {session_id} Counter has been increased by 1。")
        except sqlite3.Error as e:
            logging.error(f"Increase session {session_id} Database error occurred when resetting counter: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def get_counter(self, session_id):
        """
        Get specified session ID Message counter value of。

        Args:
            session_id (str): Session ID。

        Returns:
            int: Session ID Corresponding message counter value。If the session ID Does not exist，Then return 0。
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT count FROM message_counts WHERE session_id = ?", (session_id,)
            )
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                return 0  # Session ID Does not exist，Return 0
        except sqlite3.Error as e:
            logging.error(f"Get session {session_id} Database error occurred when resetting counter: {e}")
            return 0  # Return when an error occurs 0，Or consider throwing an exception，Decide based on specific needs
        finally:
            if conn:
                conn.close()

    def adjust_counter_if_necessary(self, session_id, context_history):
        """
        Check if the context history dialogue round length is less than the message counter，Adjust the counter if it is less。

        Args:
            session_id (str): Session ID。
            context_history (list): Context history dialogue list of the large model。
                                    Assume context_history Is a message list，
                                    Each message represents a user or AI A speech。
                                    Round length can be simply understood as the length of the message list。
        """
        current_counter = self.get_counter(session_id)
        history_length = len(context_history)

        if history_length < current_counter:
            logging.warning(
                f"Unexpected situation: Session {session_id} Context history length of ({history_length}) Less than the message counter ({current_counter})，There may be data inconsistency。"
            )
            conn = None
            try:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE message_counts SET count = ? WHERE session_id = ?",
                    (history_length, session_id),
                )
                conn.commit()
                logging.warning(f"Counter has been adjusted to context history length ({history_length})。")
                return False
            except sqlite3.Error as e:
                logging.error(f"Adjust session {session_id} Database error occurred when resetting counter: {e}")
                if conn:
                    conn.rollback()
                return False  # Return even if adjustment fails False，Indicates that further processing may be needed
            finally:
                if conn:
                    conn.close()
        else:
            logging.debug(
                f"Session {session_id} Context history length of ({history_length}) With the message counter ({current_counter}) Consistent。"
            )
            return True
