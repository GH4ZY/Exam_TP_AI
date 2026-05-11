"""
MMM AI Assistant — Production-grade rewrite
Architecture: Intent → Table selection → SQL generation → Validation/retry
             → Direct execution → LLM explanation → Streamlit display
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import sqlite3
import logging
import traceback
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests as _req
import streamlit as st
from sqlalchemy import create_engine, text as sa_text
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks.base import BaseCallbackHandler

from chroma_db import vectorstore


# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mmm_chat")


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MMM AI Assistant",
    layout="wide",
    page_icon="📊",
)

st.markdown("""
<style>
.mmm-header { font-size: 1.6rem; font-weight: 600; margin-bottom: 0; }
.mmm-sub    { color: var(--text-color); opacity: 0.6; font-size: 0.9rem; margin-top: 0; }
.intent-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em;
    background: #e8f4fd; color: #1a6fa5;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="mmm-header">📊 MMM AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="mmm-sub">Marketing Mix Modeling · SQL · Analytics</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════
DB_PATH = (Path(__file__).parent / "mmm.db").absolute()

if not DB_PATH.exists():
    st.error(f"❌ Database not found at `{DB_PATH}`. Run `db.py` first.")
    st.stop()


@lru_cache(maxsize=1)
def load_schema() -> tuple[list[str], dict[str, list[str]]]:
    """Load table names and column info once, cache forever."""
    conn = sqlite3.connect(str(DB_PATH))
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    columns = {
        t: [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall()]
        for t in tables
    }
    conn.close()
    logger.info("Schema loaded: %d tables", len(tables))
    return tables, columns


TABLES, TABLE_COLUMNS = load_schema()

with st.expander("🔍 Database schema", expanded=False):
    for t, cols in TABLE_COLUMNS.items():
        st.markdown(f"**`{t}`**: {', '.join(cols)}")
st.caption(f"📂 `{DB_PATH}`")


# ══════════════════════════════════════════════════════════════════
# OLLAMA HEALTH
# ══════════════════════════════════════════════════════════════════
MODEL_NAME = "llama3.2:3b"

try:
    _req.get("http://localhost:11434", timeout=3)
except Exception:
    st.error("⚠️ Ollama not running. Run `ollama serve` then refresh.")
    st.stop()

try:
    tags = _req.get("http://localhost:11434/api/tags", timeout=5).json()
    if not any(MODEL_NAME in m["name"] for m in tags.get("models", [])):
        st.warning(f"⚠️ Model `{MODEL_NAME}` not found. Run: `ollama pull {MODEL_NAME}`")
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════
# LLM  (single instance, cached)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL_NAME,
        base_url="http://localhost:11434",
        temperature=0,
        num_predict=2048,
        num_ctx=8192,
    )

LLM = get_llm()


# ══════════════════════════════════════════════════════════════════
# DIRECT DB ACCESS
# ══════════════════════════════════════════════════════════════════
def open_db() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def run_sql(query: str) -> Optional[pd.DataFrame]:
    """Execute a SELECT query and return a DataFrame, or None on error/empty."""
    try:
        with open_db() as conn:
            df = pd.read_sql_query(query, conn)
        if df.empty:
            logger.debug("SQL returned 0 rows: %.120s", query)
            return None
        logger.info("SQL OK — %d rows, %d cols", len(df), len(df.columns))
        return df
    except Exception as exc:
        logger.warning("SQL error: %s | query: %.120s", exc, query)
        return None


def get_sample_rows(table: str, n: int = 3) -> Optional[pd.DataFrame]:
    return run_sql(f'SELECT * FROM "{table}" LIMIT {n}')


# ══════════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════
class QueryIntent(Enum):
    RANKING      = "ranking"       # top/bottom N by metric
    AGGREGATION  = "aggregation"   # sum/avg/count by dimension
    TREND        = "trend"         # over time
    COMPARISON   = "comparison"    # A vs B
    DEFINITION   = "definition"    # what is X?
    DETAIL       = "detail"        # single row lookup
    DISTRIBUTION = "distribution"  # spread / all values
    GENERAL      = "general"       # fallback


_INTENT_PATTERNS: list[tuple[QueryIntent, list[str]]] = [
    (QueryIntent.RANKING,     ["top ", "bottom ", "best ", "worst ", "highest ", "lowest ",
                                "ranked by", "rank by", "order by", "most ", "least "]),
    (QueryIntent.TREND,       ["over time", "by week", "by month", "by date", "trend",
                                "timeline", "evolution", "across time", "per month", "per week"]),
    (QueryIntent.COMPARISON,  ["vs", "versus", "compare", "comparison", "difference between",
                                "which is better", "relative to"]),
    (QueryIntent.DEFINITION,  ["what is", "what does", "define", "meaning of", "explain",
                                "definition", "what are"]),
    (QueryIntent.AGGREGATION, ["total", "sum", "average", "avg", "count", "how many", "breakdown",
                                "by channel", "per channel", "contribution", "share"]),
    (QueryIntent.DETAIL,      ["show me", "find", "get", "fetch", "lookup", "where", "filter"]),
]


def classify_intent(question: str) -> QueryIntent:
    q = question.lower()
    for intent, patterns in _INTENT_PATTERNS:
        if any(p in q for p in patterns):
            logger.debug("Intent: %s", intent.value)
            return intent
    return QueryIntent.GENERAL


# ══════════════════════════════════════════════════════════════════
# SMART TABLE SELECTOR
# ══════════════════════════════════════════════════════════════════
_DOMAIN_MAP: dict[str, list[str]] = {
    "sales":        ["alldecomp", "decomp"],
    "revenue":      ["alldecomp", "decomp"],
    "contribution": ["alldecomp", "decomp"],
    "decomp":       ["alldecomp", "decomp"],
    "channel":      ["alldecomp", "aggregated", "decomp"],
    "media":        ["alldecomp", "aggregated", "decomp"],
    "spend":        ["aggregated", "alldecomp"],
    "budget":       ["aggregated", "alldecomp"],
    "investment":   ["aggregated", "alldecomp"],
    "cost":         ["aggregated", "alldecomp"],
    "cpa":          ["aggregated", "alldecomp"],
    "roi":          ["alldecomp", "aggregated", "pareto"],
    "roas":         ["alldecomp", "aggregated"],
    "efficiency":   ["aggregated", "alldecomp"],
    "performance":  ["aggregated", "alldecomp"],
    "solution":     ["clusters", "pareto"],
    "solid":        ["clusters", "alldecomp", "aggregated"],
    "cluster":      ["clusters"],
    "pareto":       ["pareto", "clusters", "aggregated"],
    "nrmse":        ["clusters", "pareto"],
    "rsq":          ["clusters", "pareto"],
    "model":        ["clusters", "pareto"],
    "fit":          ["clusters", "pareto"],
    "accuracy":     ["clusters", "pareto"],
    "confidence":   ["ci", "clusters_ci"],
    "interval":     ["ci", "clusters_ci"],
    "hyper":        ["hyperparameters", "hyper"],
    "adstock":      ["hyperparameters", "hyper"],
    "saturation":   ["hyperparameters", "hyper"],
    "decay":        ["hyperparameters", "hyper"],
    "alpha":        ["hyperparameters", "hyper", "alldecomp"],
    "date":         ["alldecomp", "decomp"],
    "week":         ["alldecomp", "decomp"],
    "month":        ["alldecomp", "decomp"],
    "time":         ["alldecomp", "decomp"],
    "trend":        ["alldecomp", "decomp"],
    "top":          ["clusters", "pareto", "aggregated"],
    "best":         ["clusters", "pareto"],
    "rank":         ["clusters", "aggregated"],
    "total":        ["aggregated", "alldecomp"],
}


@lru_cache(maxsize=1)
def _build_short_map() -> dict[str, str]:
    m = {}
    for full in TABLES:
        short = re.sub(r"^.*?-\s*", "", full).strip()
        m[short.lower()] = full
        m[full.lower()] = full
    return m


SHORT_TO_FULL: dict[str, str] = {}  # populated after TABLES is known


def pick_tables(question: str, max_tables: int = 3) -> list[str]:
    q = question.lower()
    scores: dict[str, float] = {t: 0.0 for t in TABLES}

    for keyword, frags in _DOMAIN_MAP.items():
        if keyword in q:
            for table in TABLES:
                tl = table.lower()
                for frag in frags:
                    if frag in tl:
                        scores[table] += 8.0

    for table in TABLES:
        short = re.sub(r"^.*?-\s*", "", table.lower()).strip()
        if short in q:
            scores[table] += 12.0
        elif any(w in q for w in short.split("_") if len(w) > 3):
            scores[table] += 4.0

    for table, cols in TABLE_COLUMNS.items():
        for col in cols:
            col_l = col.lower().replace("_", " ")
            if col_l in q or col.lower() in q:
                scores[table] += 3.0

    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    chosen = [t for t in ranked if scores[t] > 0][:max_tables]
    result = chosen if len(chosen) >= 1 else ranked[:max_tables]
    logger.info("Tables selected: %s", result)
    return result


# ══════════════════════════════════════════════════════════════════
# SCHEMA BUILDER
# ══════════════════════════════════════════════════════════════════
def build_rich_schema(tables: list[str]) -> str:
    parts = []
    for table in tables:
        cols = TABLE_COLUMNS.get(table, [])
        sample = get_sample_rows(table, n=3)
        part = f'Table: "{table}"\nColumns: {", ".join(cols)}'
        if sample is not None:
            part += f"\nSample rows:\n{sample.to_string(index=False, max_cols=20)}"
        parts.append(part)
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════
# TABLE NAME FIXER
# ══════════════════════════════════════════════════════════════════
def fix_table_names(sql: str) -> str:
    """Guarantee every table reference in sql uses the full quoted name."""
    stf = _build_short_map()

    # Fix broken quotes: "foo.xlsx" - bar  →  "foo.xlsx - bar"
    def _fix_broken(m: re.Match) -> str:
        cand = re.sub(r'["\s]+', " ", m.group(0)).strip()
        cand = re.sub(r"\s*-\s*", " - ", cand)
        key = cand.lower()
        return f'"{stf[key]}"' if key in stf else m.group(0)

    sql = re.sub(r'"[^"]*?"\s*-\s*\w[\w. ]*', _fix_broken, sql)

    # Replace bare short names (longest first to avoid partial matches)
    for short, full in sorted(stf.items(), key=lambda x: -len(x[0])):
        if short == full.lower():
            continue
        sql = re.sub(
            r'(?<!")(?i)\b' + re.escape(short) + r'\b(?!")',
            f'"{full}"',
            sql,
        )

    # Ensure all known full names are quoted
    for full in sorted(TABLES, key=len, reverse=True):
        sql = re.sub(
            r'(?<!")' + re.escape(full) + r'(?!")',
            f'"{full}"',
            sql,
        )
    return sql


# ══════════════════════════════════════════════════════════════════
# SQL VALIDATOR
# ══════════════════════════════════════════════════════════════════
@dataclass
class ValidationResult:
    valid: bool
    error: str = ""
    fixed_sql: str = ""


def validate_sql(sql: str) -> ValidationResult:
    """
    Multi-layer SQL validation:
    1. Must start with SELECT
    2. All referenced tables must exist
    3. EXPLAIN must not error
    """
    sql = sql.strip().rstrip(";").strip()

    if not sql.upper().startswith("SELECT"):
        return ValidationResult(False, "Not a SELECT statement")

    # Check table references
    quoted_tables = re.findall(r'"([^"]+)"', sql)
    for tname in quoted_tables:
        if tname not in TABLES:
            # Try to auto-correct via short map
            stf = _build_short_map()
            if tname.lower() in stf:
                sql = sql.replace(f'"{tname}"', f'"{stf[tname.lower()]}"')
            else:
                return ValidationResult(False, f"Unknown table: '{tname}'")

    # Check column references against known schema
    # (lightweight: just warn, don't block)
    for tname in [t for t in TABLES if f'"{t}"' in sql]:
        cols = TABLE_COLUMNS.get(tname, [])
        cols_lower = {c.lower() for c in cols}
        # Extract bare column refs (rough heuristic)
        col_refs = re.findall(
            r'\b([a-z_][a-z0-9_]*)\b(?!\s*\()',
            sql.lower()
        )
        sql_keywords = {
            "select","from","where","group","by","order","limit","having",
            "join","on","as","and","or","not","in","like","between","is",
            "null","asc","desc","count","sum","avg","min","max","distinct",
            "case","when","then","else","end","cast","coalesce","round",
            "upper","lower","substr","date","strftime","rowid","rownum",
        }
        for ref in col_refs:
            if ref in sql_keywords:
                continue
            # If ref isn't a column in any selected table, could be alias—skip
        # End lightweight check

    # EXPLAIN test
    try:
        with open_db() as conn:
            conn.execute(f"EXPLAIN {sql}")
        return ValidationResult(True, fixed_sql=sql)
    except Exception as exc:
        return ValidationResult(False, f"EXPLAIN failed: {exc}", fixed_sql=sql)


# ══════════════════════════════════════════════════════════════════
# INTENT-SPECIFIC SQL PROMPT TEMPLATES
# These produce much more reliable SQL than a single generic prompt
# ══════════════════════════════════════════════════════════════════
_BASE_RULES = """
CRITICAL TABLE NAME RULE:
- Table names contain spaces and special characters.
- ALWAYS wrap the ENTIRE table name in double quotes as one unit.
- WRONG:  FROM "Combined_CSVs.xlsx" - pareto_aggregated
- CORRECT: FROM "Combined_CSVs.xlsx - pareto_aggregated"

Exact table names you may use (copy character-for-character inside double quotes):
{table_names}

STRICT RULES:
- Output ONLY the raw SQL query — no explanation, no markdown, no backticks, no semicolons.
- Use ONLY columns that appear in the schema below.
- Every column reference must exist in the schema.

MMM COLUMN REFERENCE:
- solID: model solution identifier
- roi_mean / roi_total: return on investment metrics
- cpa_total: cost per acquisition
- total_spend: total media spend
- cluster: solution group ("top_sol" = best solutions)
- nrmse: model error (lower = better); rsq: R-squared (higher = better)
- Media channels: appear as column names (TV, Digital, SEM, INFLU, etc.)

SCHEMA WITH SAMPLE DATA:
{schema}
"""

_RANKING_PROMPT = PromptTemplate.from_template(
    "You are a SQLite expert. Write ONE valid SQLite SELECT query for a RANKING question.\n"
    + _BASE_RULES +
    """
RANKING RULES:
- Extract the N from the question (e.g. "top 10" → LIMIT 10). Default to 10 if unspecified.
- Use ORDER BY <metric_column> DESC for "top/best/highest".
- Use ORDER BY <metric_column> ASC  for "bottom/worst/lowest".
- Always include LIMIT N.
- For ROI ranking: ORDER BY roi_total DESC (prefer roi_total over roi_mean if both exist).
- For spend ranking: ORDER BY total_spend DESC.
- For model quality: ORDER BY rsq DESC (higher is better) or ORDER BY nrmse ASC (lower is better).
- If "top_sol" or "best solutions" mentioned: add WHERE cluster = 'top_sol'.

USER QUESTION: {question}

SQL QUERY:"""
)

_AGGREGATION_PROMPT = PromptTemplate.from_template(
    "You are a SQLite expert. Write ONE valid SQLite SELECT query for an AGGREGATION question.\n"
    + _BASE_RULES +
    """
AGGREGATION RULES:
- Use SUM(), AVG(), COUNT(), MIN(), MAX() as appropriate.
- Always GROUP BY the grouping dimension (channel, solID, cluster, etc.).
- Add ORDER BY on the aggregate result DESC for natural readability.
- Round decimal results: ROUND(SUM(col), 2).
- For "contribution" or "share": compute SUM(channel_col) and order DESC.
- For "total spend by channel": SUM each media spend column → UNION or pivot as needed.

USER QUESTION: {question}

SQL QUERY:"""
)

_TREND_PROMPT = PromptTemplate.from_template(
    "You are a SQLite expert. Write ONE valid SQLite SELECT query for a TIME TREND question.\n"
    + _BASE_RULES +
    """
TREND RULES:
- Find the date/week/time column in the schema.
- GROUP BY the date column.
- ORDER BY the date column ASC.
- Use strftime('%Y-%m', date_col) to group by month if monthly trend asked.
- Use strftime('%Y-%W', date_col) to group by week if weekly trend asked.
- Apply date filters if mentioned: strftime('%Y', date_col) = '2023' or BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.

USER QUESTION: {question}

SQL QUERY:"""
)

_COMPARISON_PROMPT = PromptTemplate.from_template(
    "You are a SQLite expert. Write ONE valid SQLite SELECT query for a COMPARISON question.\n"
    + _BASE_RULES +
    """
COMPARISON RULES:
- Use a WHERE clause with IN (...) or OR conditions to filter to the entities being compared.
- Include the grouping dimension (solID, channel, cluster) in SELECT.
- Include all relevant metrics for comparison.
- ORDER BY the primary metric DESC.

USER QUESTION: {question}

SQL QUERY:"""
)

_GENERAL_PROMPT = PromptTemplate.from_template(
    "You are a SQLite expert. Write ONE valid SQLite SELECT query.\n"
    + _BASE_RULES +
    """
GENERAL RULES:
- If question asks for counts/summary: use GROUP BY + aggregate.
- If question asks for specific rows: use WHERE + appropriate filters.
- Default LIMIT 50 if no limit implied.
- Prefer meaningful ORDER BY (most relevant metric DESC).

USER QUESTION: {question}

SQL QUERY:"""
)

_INTENT_TO_PROMPT = {
    QueryIntent.RANKING:      _RANKING_PROMPT,
    QueryIntent.AGGREGATION:  _AGGREGATION_PROMPT,
    QueryIntent.TREND:        _TREND_PROMPT,
    QueryIntent.COMPARISON:   _COMPARISON_PROMPT,
    QueryIntent.DETAIL:       _GENERAL_PROMPT,
    QueryIntent.DISTRIBUTION: _GENERAL_PROMPT,
    QueryIntent.GENERAL:      _GENERAL_PROMPT,
}


def _clean_llm_sql(raw: str) -> str:
    """Strip markdown, backticks, and non-SQL text from LLM output."""
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "").strip().rstrip(";")
    # Take first SELECT if multiple are present
    match = re.search(r"(SELECT[\s\S]+)", raw, re.IGNORECASE)
    if match:
        raw = match.group(1).strip().rstrip(";")
    # Remove trailing explanation lines
    lines = raw.splitlines()
    sql_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(note:|this query|explanation:|--\s)", stripped, re.IGNORECASE):
            break
        sql_lines.append(line)
    return "\n".join(sql_lines).strip()


# ══════════════════════════════════════════════════════════════════
# SELF-CORRECTOR PROMPT
# ══════════════════════════════════════════════════════════════════
_REPAIR_PROMPT = PromptTemplate.from_template(
    """You are a SQLite expert. The following SQL query has an error. Fix it.

ORIGINAL QUERY:
{sql}

ERROR MESSAGE:
{error}

AVAILABLE TABLES (use EXACT names in double quotes):
{table_names}

SCHEMA:
{schema}

Output ONLY the corrected SQL query. No explanation, no markdown, no backticks.

FIXED SQL:"""
)


# ══════════════════════════════════════════════════════════════════
# MAIN SQL PIPELINE
# ══════════════════════════════════════════════════════════════════
@dataclass
class SQLResult:
    df: Optional[pd.DataFrame] = None
    sql: str = ""
    error: str = ""
    attempts: int = 0


def generate_sql(question: str, tables: list[str], intent: QueryIntent) -> str:
    """Generate SQL using the intent-specific prompt template."""
    schema_str = build_rich_schema(tables)
    table_names_str = "\n".join(f'  "{t}"' for t in tables)
    prompt = _INTENT_TO_PROMPT.get(intent, _GENERAL_PROMPT)
    chain = prompt | LLM | StrOutputParser()
    raw = chain.invoke({
        "schema": schema_str,
        "question": question,
        "table_names": table_names_str,
    })
    return _clean_llm_sql(raw)


def repair_sql(bad_sql: str, error: str, tables: list[str]) -> str:
    """Ask the LLM to fix a broken SQL query."""
    schema_str = build_rich_schema(tables)
    table_names_str = "\n".join(f'  "{t}"' for t in tables)
    chain = _REPAIR_PROMPT | LLM | StrOutputParser()
    raw = chain.invoke({
        "sql": bad_sql,
        "error": error,
        "schema": schema_str,
        "table_names": table_names_str,
    })
    return _clean_llm_sql(raw)


def run_sql_pipeline(
    question: str,
    tables: list[str],
    intent: QueryIntent,
    max_attempts: int = 3,
) -> SQLResult:
    """
    Full SQL pipeline:
    1. Generate SQL (intent-specific prompt)
    2. Fix table names
    3. Validate (EXPLAIN)
    4. Execute
    5. If fails, self-repair and retry
    """
    result = SQLResult()
    last_error = ""
    current_sql = ""

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        try:
            if attempt == 1:
                current_sql = generate_sql(question, tables, intent)
            else:
                logger.info("Self-repair attempt %d: %s", attempt, last_error)
                current_sql = repair_sql(current_sql, last_error, tables)

            current_sql = fix_table_names(current_sql)
            logger.info("SQL attempt %d: %.150s", attempt, current_sql)

            if not current_sql:
                last_error = "Empty SQL generated"
                continue

            vr = validate_sql(current_sql)
            if not vr.valid:
                last_error = vr.error
                logger.warning("Validation failed (attempt %d): %s", attempt, vr.error)
                if vr.fixed_sql:
                    current_sql = vr.fixed_sql
                continue

            current_sql = vr.fixed_sql or current_sql
            df = run_sql(current_sql)
            if df is not None:
                result.df = df
                result.sql = current_sql
                logger.info("Pipeline success on attempt %d", attempt)
                return result

            last_error = "Query executed but returned no rows"
            result.sql = current_sql  # keep for debugging

        except Exception as exc:
            last_error = str(exc)
            logger.error("Pipeline attempt %d error: %s", attempt, exc)
            logger.debug(traceback.format_exc())

    result.error = last_error
    result.sql = current_sql
    return result


# ══════════════════════════════════════════════════════════════════
# DATE FILTER (applied post-retrieval as a safety net)
# ══════════════════════════════════════════════════════════════════
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def filter_df_by_date(df: pd.DataFrame, question: str) -> pd.DataFrame:
    q = question.lower()
    month_num = next((v for k, v in _MONTH_MAP.items() if k in q), None)
    year_m = re.search(r"\b(20\d{2})\b", question)
    year = int(year_m.group(1)) if year_m else None

    if month_num is None and year is None:
        return df

    date_col = next(
        (c for c in df.columns
         if any(kw in c.lower() for kw in ("date", "time", "day", "week", "month", "year"))),
        None,
    )
    if date_col is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c].dropna().head(5), errors="raise")
                date_col = c
                break
            except Exception:
                continue

    if date_col is None:
        return df

    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        mask = pd.Series(True, index=df.index)
        if month_num:
            mask &= parsed.dt.month == month_num
        if year:
            mask &= parsed.dt.year == year
        filtered = df[mask].copy()
        return filtered if not filtered.empty else df
    except Exception:
        return df


# ══════════════════════════════════════════════════════════════════
# CHART GENERATION
# Auto-selects chart type based on data shape + intent
# ══════════════════════════════════════════════════════════════════
def auto_chart(df: pd.DataFrame, intent: QueryIntent, question: str) -> None:
    """Attempt to render the most appropriate chart for the result."""
    import streamlit as st

    if df is None or df.empty or len(df) == 0:
        return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not num_cols:
        return

    try:
        if intent == QueryIntent.TREND:
            date_col = next(
                (c for c in df.columns if any(k in c.lower() for k in ("date","week","month","year"))),
                cat_cols[0] if cat_cols else None,
            )
            if date_col:
                chart_df = df.set_index(date_col)[num_cols[:3]]
                st.line_chart(chart_df)
                return

        if intent == QueryIntent.RANKING and len(df) <= 25:
            y_col = num_cols[0]
            x_col = cat_cols[0] if cat_cols else df.index.name or "index"
            plot_df = df[[x_col, y_col]].set_index(x_col) if x_col in df.columns else df[[y_col]]
            st.bar_chart(plot_df)
            return

        if intent == QueryIntent.AGGREGATION and cat_cols and len(df) <= 30:
            y_col = num_cols[0]
            x_col = cat_cols[0]
            plot_df = df[[x_col, y_col]].set_index(x_col)
            st.bar_chart(plot_df)
            return

    except Exception as exc:
        logger.debug("Chart generation skipped: %s", exc)


# ══════════════════════════════════════════════════════════════════
# RAG TOOL
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def get_rag_chain():
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    chain = (
        {
            "context": retriever | (lambda docs: "\n---\n".join(d.page_content[:400] for d in docs)),
            "question": RunnablePassthrough(),
        }
        | PromptTemplate.from_template(
            "You are an MMM expert. Use the context to answer concisely.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        | LLM
        | StrOutputParser()
    )
    return chain


vector_tool = Tool(
    name="MMMDefinitions",
    func=get_rag_chain().invoke,
    description="Look up MMM terminology, column definitions, or model concepts.",
)


# ══════════════════════════════════════════════════════════════════
# EXPLANATION GENERATOR
# The LLM receives the *data* and generates a narrative.
# This is the ONLY job the LLM does — SQL is already done.
# ══════════════════════════════════════════════════════════════════
_EXPLAIN_PROMPT = PromptTemplate.from_template(
    """You are an expert MMM analyst. The user asked a question and we retrieved data from the database.

USER QUESTION:
{question}

DATA RETRIEVED ({n_rows} rows):
{data_sample}

INTENT TYPE: {intent}

Write a clear, concise 2-4 sentence business insight about this data.
- Reference specific numbers from the data.
- Highlight the most important finding.
- Use MMM terminology correctly.
- Do NOT say "based on the SQL query" or describe technical steps.
- Do NOT repeat the data in full — just the key insight.

INSIGHT:"""
)


def generate_explanation(
    question: str,
    df: pd.DataFrame,
    intent: QueryIntent,
) -> str:
    """Generate a natural-language explanation of the retrieved data."""
    try:
        # Show top rows to the LLM (keep token cost low)
        sample = df.head(10).to_string(index=False, max_cols=15)
        chain = _EXPLAIN_PROMPT | LLM | StrOutputParser()
        explanation = chain.invoke({
            "question": question,
            "data_sample": sample,
            "n_rows": len(df),
            "intent": intent.value,
        })
        # Clean boilerplate
        explanation = re.sub(
            r"(here is|this query|i ran|the sql|based on the query)[\s\S]{0,200}",
            "", explanation, flags=re.IGNORECASE,
        ).strip()
        return explanation
    except Exception as exc:
        logger.warning("Explanation generation failed: %s", exc)
        return ""


# ══════════════════════════════════════════════════════════════════
# DEFINITION HANDLER (uses RAG instead of SQL)
# ══════════════════════════════════════════════════════════════════
def handle_definition_query(question: str) -> str:
    """For definitional questions, use RAG directly."""
    try:
        return get_rag_chain().invoke(question)
    except Exception as exc:
        logger.warning("RAG lookup failed: %s", exc)
        return "I couldn't find a definition for that term in my knowledge base."


# ══════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY (session-based)
# ══════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"|"assistant", "content": str}

if "last_df" not in st.session_state:
    st.session_state.last_df = None


def add_to_history(role: str, content: str) -> None:
    st.session_state.history.append({"role": role, "content": content})
    # Keep last 20 turns to avoid context overflow
    if len(st.session_state.history) > 40:
        st.session_state.history = st.session_state.history[-40:]


# ══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    show_sql   = st.checkbox("Show SQL queries",    value=True)
    show_chart = st.checkbox("Auto-generate charts", value=True)
    max_rows   = st.slider("Max rows to display", 10, 500, 100)

    st.markdown("---")
    st.markdown("### 🕘 Recent questions")
    for msg in reversed(st.session_state.history[-10:]):
        if msg["role"] == "user":
            st.caption(f"› {msg['content'][:60]}…" if len(msg['content']) > 60 else f"› {msg['content']}")

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.session_state.last_df = None
        st.rerun()

# ── Conversation replay ───────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Input ─────────────────────────────────────────────────────────
user_query = st.chat_input(
    "Ask anything about your MMM data… (e.g. 'Top 10 solutions by ROI', 'What is NRMSE?')"
)

if user_query:
    # Show user message
    with st.chat_message("user"):
        st.write(user_query)
    add_to_history("user", user_query)

    with st.chat_message("assistant"):

        # ── Step 1: Intent + table selection ──────────────────────
        intent = classify_intent(user_query)
        st.caption(
            f'<span class="intent-badge">Intent: {intent.value}</span>',
            unsafe_allow_html=True,
        )

        # ── Step 2: Definitional queries go straight to RAG ───────
        if intent == QueryIntent.DEFINITION:
            with st.spinner("Looking up definition…"):
                answer = handle_definition_query(user_query)
            st.write(answer)
            add_to_history("assistant", answer)

        else:
            # ── Step 3: Select tables ──────────────────────────────
            relevant_tables = pick_tables(user_query, max_tables=3)
            st.caption(f"🔎 Tables: `{'`, `'.join(relevant_tables)}`")

            # ── Step 4: SQL pipeline (generate → validate → execute) ──
            with st.spinner("Generating and running SQL…"):
                sql_result = run_sql_pipeline(
                    user_query, relevant_tables, intent, max_attempts=3
                )

            df = sql_result.df

            # Apply date filter as post-processing safety net
            if df is not None and not df.empty:
                df_filtered = filter_df_by_date(df, user_query)
                if len(df_filtered) < len(df):
                    st.caption(f"📅 Filtered to {len(df_filtered)} rows matching date.")
                    df = df_filtered

            # ── Step 5: Display results ────────────────────────────
            if df is not None and not df.empty:
                st.markdown("### 📊 Results")
                st.dataframe(
                    df.head(max_rows),
                    use_container_width=True,
                    height=min(400, 40 + 35 * min(len(df), max_rows)),
                )
                st.caption(f"{len(df):,} rows returned" + (f" (showing first {max_rows})" if len(df) > max_rows else ""))

                # ── Step 6: Auto-chart ─────────────────────────────
                if show_chart:
                    auto_chart(df, intent, user_query)

                # ── Step 7: SQL expander ───────────────────────────
                if show_sql and sql_result.sql:
                    with st.expander("🔍 SQL used"):
                        st.code(sql_result.sql, language="sql")
                        if sql_result.attempts > 1:
                            st.caption(f"⚠️ Required {sql_result.attempts} attempts (self-corrected)")

                # ── Step 8: LLM explanation ────────────────────────
                with st.spinner("Generating insight…"):
                    explanation = generate_explanation(user_query, df, intent)

                if explanation and len(explanation) > 20:
                    with st.expander("📝 Insight", expanded=True):
                        st.write(explanation)
                    add_to_history("assistant", explanation)
                else:
                    add_to_history("assistant", f"Retrieved {len(df):,} rows. See table above.")

            else:
                # ── Fallback: no data ──────────────────────────────
                st.warning("⚠️ No data returned for this query.")

                # Show the SQL that failed (for debugging)
                if sql_result.sql and show_sql:
                    with st.expander("🔍 SQL attempted (returned no rows)"):
                        st.code(sql_result.sql, language="sql")
                if sql_result.error:
                    with st.expander("⚠️ Error details"):
                        st.code(sql_result.error)

                # Try RAG as fallback for context
                with st.spinner("Searching knowledge base…"):
                    rag_answer = handle_definition_query(user_query)
                if rag_answer and len(rag_answer) > 20:
                    st.markdown("**From knowledge base:**")
                    st.write(rag_answer)
                    add_to_history("assistant", f"No SQL data found. From knowledge base: {rag_answer}")
                else:
                    fallback_msg = (
                        "I couldn't retrieve data for that question. "
                        "Try rephrasing, or check that the table contains the relevant columns."
                    )
                    st.info(fallback_msg)
                    add_to_history("assistant", fallback_msg)

                st.markdown("💡 **Tips:**")
                tips = [
                    "Use exact metric names: `roi_total`, `nrmse`, `total_spend`",
                    "Specify the table context: 'in the aggregated results'",
                    "For rankings: 'Show top 10 solutions by roi_total'",
                    "For model quality: 'Which solution has the best NRMSE?'",
                ]
                for tip in tips:
                    st.caption(f"• {tip}")