"""Read-only, schema-aware MySQL lookups for procurement agent tools."""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_community.utilities import SQLDatabase

from code.google_llm import get_google_llm


FORBIDDEN_SQL = re.compile(
    r"\b(ALTER|CREATE|DELETE|DROP|GRANT|INSERT|MERGE|RENAME|REPLACE|REVOKE|"
    r"TRUNCATE|UPDATE|USE)\b",
    flags=re.IGNORECASE,
)
TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_table_names(value: str, setting_name: str) -> list[str]:
    """Validate a comma-separated table allowlist supplied through environment settings."""
    table_names = [table_name.strip() for table_name in value.split(",") if table_name.strip()]
    if not table_names:
        raise ValueError(f"{setting_name} must list at least one MySQL table.")
    if invalid_names := [name for name in table_names if not TABLE_NAME.fullmatch(name)]:
        raise ValueError(f"Invalid MySQL table name(s): {', '.join(invalid_names)}")
    return table_names


def _validate_read_only_sql(sql: str, allowed_tables: Sequence[str]) -> str:
    """Allow a single SELECT/WITH statement that references only configured tables."""
    statement = sql.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
    if statement.endswith(";"):
        statement = statement[:-1].strip()
    if ";" in statement or not statement.upper().startswith(("SELECT", "WITH")):
        raise ValueError("The generated SQL must contain exactly one SELECT or WITH statement.")
    if FORBIDDEN_SQL.search(statement):
        raise ValueError("The generated SQL contains a non-read-only operation.")
    if "--" in statement or "/*" in statement:
        raise ValueError("The generated SQL must not contain comments.")

    referenced_tables = {
        table_name.lower()
        for table_name in re.findall(r"\b(?:FROM|JOIN)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", statement, re.I)
    }
    allowed_names = {table_name.lower() for table_name in allowed_tables}
    if not referenced_tables or not referenced_tables.issubset(allowed_names):
        raise ValueError("The generated SQL references a table outside this tool's allowlist.")
    if not re.search(r"\bLIMIT\s+\d+\b", statement, re.IGNORECASE):
        statement = f"{statement} LIMIT 100"
    return statement


def run_mysql_lookup(
    *,
    question: str,
    mysql_uri: str,
    table_names: Sequence[str],
    domain: str,
) -> str:
    """Generate and execute a safe MySQL SELECT query for one procurement domain."""
    # Limit LangChain's schema context to approved tables and include headers plus one example row.
    database = SQLDatabase.from_uri(
        mysql_uri,
        include_tables=list(table_names),
        sample_rows_in_table_info=1,
    )
    table_context = database.get_table_info(table_names=list(table_names))
    sql_prompt = f"""You write MySQL SELECT queries for {domain} data.
Use only the tables listed in the schema context. The context includes each table's
headers and one sample row; sample values are illustrative and must not be assumed
to be complete. Answer the user's request with exactly one MySQL SELECT or WITH query.
Never use data-changing statements, multiple statements, comments, or Markdown fences.
Use LIMIT 100 unless the user explicitly asks for a smaller result set.

Schema context:
{table_context}

User question: {question}
"""
    generated_sql = str(get_google_llm().invoke(sql_prompt).content)
    safe_sql = _validate_read_only_sql(generated_sql, table_names)
    # SQLDatabase executes through SQLAlchemy; validation above prevents database mutations.
    result = database.run(safe_sql)
    return f"Generated SQL:\n{safe_sql}\n\nQuery results:\n{result}"
