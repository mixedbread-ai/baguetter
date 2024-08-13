from __future__ import annotations

# ruff: noqa
import builtins
import pickle
import sqlite3
from typing import Any, Iterator

from baguetter.utils.common import batch_iter


def check_key(key: str | None) -> None:
    """Simple type checking of key."""
    if key is None:
        raise ValueError("Key cannot be None")
    return str(key)


def serialize(val: Any) -> bytes:
    """Serialize the value using pickle."""
    return pickle.dumps(val)


def deserialize(val: bytes) -> Any:
    """Deserialize the value using pickle."""
    return pickle.loads(val)


class KeyValueSqlite:
    """Acts like a python dictionary but stores values to the backing sqlite
    database file. Uses pickle for serialization to support complex Python objects.
    """

    def __init__(
        self,
        path: str = ":memory:",
        table_name: str | None = None,
    ) -> None:
        """Initialize the database."""
        table_name = table_name or "default_table"
        self.table_name = table_name.replace("-", "_")
        self.conn = sqlite3.connect(path)

        self.create_table()

    def create_table(self) -> None:
        """Creates the table if it doesn't already exist."""
        check_table_stmt = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}';"
        cursor = self.conn.execute(check_table_stmt)
        has_table = cursor.fetchall()
        if has_table:
            return
        create_stmt = f"CREATE TABLE {self.table_name} (key TEXT PRIMARY KEY UNIQUE NOT NULL, value BLOB);"
        try:
            self.conn.executescript(create_stmt)
        except sqlite3.ProgrammingError:
            pass  # Table already created

    def set_default(self, key: str, val: Any) -> None:
        """Like python dictionary set_default(), only settings the value to the default
        if the value doesn't already exist.
        """
        check_key(key)
        val = serialize(val)
        insert_stmt = f"INSERT OR IGNORE INTO {self.table_name} (key, value) VALUES (?, ?)"
        record = (key, val)
        with self.conn:
            self.conn.execute(insert_stmt, record)

    def __setitem__(self, key: str, item: Any) -> Any:
        """Same as dict.__setitem__()."""
        return self.set(key, item)

    def __getitem__(self, key: str) -> Any:
        """Same as dict.__getitem__()."""
        return self.get_or_raise(key)

    def __missing__(self, key: str) -> None:
        """Same as dict.__missing__()."""
        msg = f"Missing key {key}"
        raise KeyError(msg)

    def __iter__(self) -> Any:
        """Same as dict.__iter__()."""
        return iter(self.to_dict())

    def __contains__(self, key: str) -> bool:
        """Same as dict.__contains__()."""
        return self.has_key(key)

    def __delitem__(self, key: str) -> None:
        """Same as dict.__delitem__()."""
        self.remove(key)

    def items(self) -> Any:
        """Same as dict.items()."""
        return self.to_dict().items()

    def __len__(self) -> int:
        """Same as dict.len()."""
        return len(self.items())

    def __repr__(self) -> str:
        """Allows a string representation."""
        out = self.to_dict()
        return f"KeyValueSqlite({out})"

    def __str__(self) -> str:
        """Allows a string representation."""
        return self.__repr__()

    def set(self, key: str, val: Any) -> None:
        """Like dict.set(key) = value."""
        check_key(key)
        val = serialize(val)
        insert_stmt = f"INSERT OR REPLACE INTO {self.table_name} (key, value) VALUES (?, ?)"
        record = (key, val)
        with self.conn:
            self.conn.execute(insert_stmt, record)

    def set_many(self, a_dict: dict[str, Any], batch_size: int = 10000) -> None:
        """Like dict.update()."""
        for key in a_dict:
            check_key(key)
        with self.conn:
            for batch in batch_iter(list(a_dict.items()), batch_size):
                insert_stmt = f"INSERT OR REPLACE INTO {self.table_name} (key, value) VALUES (?, ?)"
                records = [(key, serialize(val)) for key, val in batch]
                self.conn.executemany(insert_stmt, records)

    def get(self, key: str | None, default: Any = None) -> Any:
        """Like dict.get(key, default)."""
        check_key(key)
        select_stmt = f"SELECT value FROM {self.table_name} WHERE (key = ?)"
        values = (key,)
        cursor = self.conn.execute(select_stmt, values)
        row = cursor.fetchone()
        if row:
            return deserialize(row[0])
        return default

    def get_or_raise(self, key: str) -> Any:
        """Returns the value if it exists, or throws a KeyError."""
        check_key(key)
        select_stmt = f"SELECT value FROM {self.table_name} WHERE (key = ?)"
        values = (key,)
        cursor = self.conn.execute(select_stmt, values)
        row = cursor.fetchone()
        if row:
            return deserialize(row[0])
        msg = f'Missing key: "{key}"'
        raise KeyError(msg)

    def has_key(self, key: str) -> bool:
        """Returns true if the key exists."""
        check_key(key)
        select_stmt = f"SELECT 1 FROM {self.table_name} WHERE (key = ?)"
        values = (key,)
        cursor = self.conn.execute(select_stmt, values)
        return cursor.fetchone() is not None

    def keys(self) -> list[str]:
        """Returns a list of keys."""
        select_stmt = f"SELECT key FROM {self.table_name}"
        cursor = self.conn.execute(select_stmt)
        return [row[0] for row in cursor]

    def key_range(self, key_low: str, key_high: str) -> list[str]:
        """Get keys between key_low and key_high."""
        check_key(key_low)
        check_key(key_high)
        select_stmt = f"SELECT key FROM {self.table_name} WHERE key BETWEEN ? AND ?"
        values = (key_low, key_high)
        cursor = self.conn.execute(select_stmt, values)
        return [row[0] for row in cursor]

    def get_many(self, a_set: builtins.set[str], batch_size: int = 10000) -> dict[str, Any]:
        """Given the set of keys, return a dictionary matching the keys to the
        values.
        """
        result = {}
        with self.conn:
            for batch in batch_iter(list(a_set), batch_size):
                select_stmt = f"SELECT key, value FROM {self.table_name} WHERE key IN ({','.join(['?'] * len(batch))})"
                cursor = self.conn.execute(select_stmt, batch)
                result.update({key: deserialize(value) for key, value in cursor})
        return result

    def dict_range(self, key_low: str, key_high: str) -> dict[str, Any]:
        """Returns a dictionary of keys to values."""
        check_key(key_low)
        check_key(key_high)
        select_stmt = f"SELECT key, value FROM {self.table_name} WHERE key BETWEEN ? AND ?"
        cursor = self.conn.execute(select_stmt, (key_low, key_high))
        return {key: deserialize(value) for key, value in cursor}

    def get_range(self, key_low: str, key_high: str) -> list[tuple[str, Any]]:
        """Outputs an ordered sequence starting from key_low to key_high."""
        check_key(key_low)
        check_key(key_high)
        select_stmt = f"SELECT key, value FROM {self.table_name} WHERE key BETWEEN ? AND ?"
        cursor = self.conn.execute(select_stmt, (key_low, key_high))
        return [(key, deserialize(value)) for key, value in cursor]

    def remove(self, key: str, ignore_missing_key: bool = False) -> None:
        """Removes the key, if it exists."""
        check_key(key)
        delete_stmt = f"DELETE FROM {self.table_name} WHERE key=?"
        with self.conn:
            cursor = self.conn.execute(delete_stmt, (key,))
            if cursor.rowcount == 0 and not ignore_missing_key:
                raise KeyError(key)

    def remove_many(
        self,
        keys: list[str],
        ignore_missing_keys: bool = False,
        batch_size: int = 10000,
    ) -> None:
        """Removes multiple keys, if they exist."""
        for key in keys:
            check_key(key)
        with self.conn:
            for batch in batch_iter(keys, batch_size):
                delete_stmt = f"DELETE FROM {self.table_name} WHERE key IN ({','.join(['?'] * len(batch))})"
                cursor = self.conn.execute(delete_stmt, batch)
                if cursor.rowcount < len(batch) and not ignore_missing_keys:
                    missing_keys = set(batch) - set(self.keys())
                    raise KeyError(f"Missing keys: {missing_keys}")

    def clear(self) -> None:
        """Removes everything from this database."""
        delete_stmt = f"DELETE FROM {self.table_name}"
        with self.conn:
            self.conn.execute(delete_stmt)

    def update(self, a_dict: dict[str, Any], batch_size: int = 10000) -> None:
        """Like dict.update()."""
        for key in a_dict:
            check_key(key)
        with self.conn:
            for batch in batch_iter(list(a_dict.items()), batch_size):
                insert_stmt = f"INSERT OR REPLACE INTO {self.table_name} (key, value) VALUES (?, ?)"
                records = [(key, serialize(val)) for key, val in batch]
                self.conn.executemany(insert_stmt, records)

    def to_dict(self) -> dict:
        """Returns the whole database as a dictionary of key->value."""
        select_stmt = f"SELECT key, value FROM {self.table_name}"
        cursor = self.conn.execute(select_stmt)
        return {key: deserialize(value) for key, value in cursor}

    def insert_or_ignore(self, a_dict: dict[str, Any], batch_size: int = 10000) -> None:
        """The value is either inserted if missing, or if present, no change
        takes place.
        """
        for key in a_dict:
            check_key(key)
        with self.conn:
            for batch in batch_iter(list(a_dict.items()), batch_size):
                insert_stmt = f"INSERT OR IGNORE INTO {self.table_name} (key, value) VALUES (?, ?)"
                records = [(key, serialize(val)) for key, val in batch]
                self.conn.executemany(insert_stmt, records)

    def atomic_add(self, key: str, value: int) -> None:
        """Adds value to the value associated with key."""
        check_key(key)
        with self.conn:
            current_value = self.get(key, 0)
            new_value = current_value + value
            self.set(key, new_value)
