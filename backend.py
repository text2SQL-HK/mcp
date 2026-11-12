#region import
from __future__ import annotations

import copy
import csv
import io
import json
import os
import re
import tarfile
import time
import oracledb
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, Union

import docker
import httpx
import oracledb
from tabulate import tabulate
from threading import Lock
import base64
import re

from openai import OpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader, TextLoader
from neo4j import GraphDatabase
import numpy as np
from openai import OpenAI
#endregion
# --- 경로/환경 설정 -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
TABLES_JSON_PATH = Path(os.environ.get("TABLES_JSON_PATH") or (BASE_DIR / "metadata" / "tables1223.json"))
ORACLE_JSON_PATH = Path(os.environ.get("ORACLE_JSON_PATH") or (Path(BASE_DIR / "ORACLE_INFO.json")))
with open(ORACLE_JSON_PATH, 'r', encoding='utf-8') as file:
    ORACLE_JSON = json.load(file)

# Configure OPENAI_API_KEY in the process environment.

#region retriever
import requests

instruction = """
You are selecting database columns that are likely required to write a correct SQL query for the user question.

Return "yes" ONLY if the column is directly needed for:
1) filtering conditions (WHERE),
2) grouping/aggregation (GROUP BY / HAVING),
3) joins/keys (JOIN),
4) selected outputs (SELECT),
5) sorting (ORDER BY)

Return "no" if the column is merely background information, redundant, or only loosely related.
Be conservative: prefer "no" unless the column is clearly useful for producing SQL.
Take care of [Context].
[Context]
{context}
"""

client = OpenAI(base_url=os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8001/v1"), api_key=os.environ.get("EMBEDDING_API_KEY") or "EMPTY")
url = os.environ.get("RERANK_URL", "http://localhost:8002/rerank")

def replace_bracket_value(s: str, new_value: str) -> str:
    before, rest = s.split("[", 1)
    _, after = rest.split("]", 1)
    return f"{before}[{new_value[1:-1]}]{after}"

def rerank(query:str, columns:List[str], context):
    payload = {
        "query": query,
        "instruction": instruction.format(context=context),
        "documents": columns
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return (resp.json())["results"]

def embed(input: str):
    response = client.embeddings.create(
        model = "/workspace/models/Qwen3-Embedding-0.6B",
        input= input
    )
    return response.data[0].embedding

class GraphRAGRetriever:
    def __init__(self, driver):
        self.driver = driver
        self.W_TABLE = 0.5
        self.W_COL_MAX = 0.35
        self.W_COL_CNT = 0.15
        
    def close(self):
        self.driver.close()

    def get_relevant_schema(self, question, context):
        with self.driver.session() as session:
            embed_question = embed(question)

            raw_candidates = session.execute_read(
                self._dual_vector_search, 
                embed_question, 
                top_k_tables=5, 
                top_k_cols=20
            )
            # 2. 하이브리드 스코어링 & 랭킹
            ranked_tables = self._calculate_hybrid_scores(raw_candidates)
            print(ranked_tables)
            # 3. Adaptive Pruning (점수 급락 지점에서 자르기)
            #pruned_tables = self._adaptive_pruning(ranked_tables)
            pruned_tables = [d["table"] for d in ranked_tables][:2]
            # 4. 그래프 연결성 확인 (고립 제거 및 연결 테이블 추가)
            """final_schema = session.execute_read(
                self._ensure_connectivity, 
                pruned_tables
            )
            """
            metadata = session.execute_read(
                self._find_column,
                pruned_tables,
                question,
                context,
                top_k=7
            )
        return metadata

    @staticmethod
    def _find_column(tx, table_names, question, context, top_k):
        query = """
        //1. 각 테이블 찾기
        MATCH (t: Table)-[:HAS_COLUMN]-(col:NonJoinable)
        WHERE t.name = $table_name
        RETURN 
            t.name AS t_name,
            t.comment AS t_comment,
            col.desc AS col_desc
        """
        query2 = """
        MATCH (t: Table)-[:HAS_COLUMN]-(col:Joinable)
        WHERE t.name = $table_name
        WITH t, col, properties(col) AS props
        RETURN 
            t.name AS t_name,
            t.comment AS t_comment,
            col.desc AS col_desc,
            props[$table_name] AS example_value
        """
        desc_str_lst : List[str] = []
        for table_name in table_names:
            col_desc_lst = []
            result = tx.run(query, table_name=table_name)
            record = result.data()
            table_desc = record[0]["t_comment"]
            desc_str_lst.append(f"#Table: {table_name}\n{table_desc}\nColumns\n")
            desc_str_lst.append("[\n")
            for row in record:
                col_desc = row["col_desc"]
                col_desc_lst.append(col_desc)
            rerank_columns = rerank(question, col_desc_lst, context)
            for i in range(top_k):
                col_desc = rerank_columns[i]["document"]
                desc_str_lst.append(f" {col_desc}\n")
            joinable = tx.run(query2, table_name=table_name).data()
            for row in joinable:
                col_desc = row["col_desc"]
                col_desc = replace_bracket_value(col_desc, row["example_value"])
                desc_str_lst.append(f" {col_desc}\n")
            desc_str_lst.append("]\n")
        
        desc_str = "".join(desc_str_lst)
        return desc_str

# [Step 1] Cypher: 테이블과 컬럼을 동시에 검색하고, 컬럼은 부모 테이블로 그룹화
    @staticmethod
    def _dual_vector_search(tx, vector, top_k_tables, top_k_cols):
        query = """
        // 1. 테이블 벡터 검색 (독립 수행)
        CALL db.index.vector.queryNodes('table_index', $k_tbl, $vec)
        YIELD node AS tbl, score AS t_score
        WITH collect({tbl: tbl.name, score: t_score}) AS table_hits
        
        // 2. 컬럼 벡터 검색
        CALL db.index.vector.queryNodes('column_index', $k_col, $vec)
        YIELD node AS col, score AS c_score
        
        // 3. 관계 확장 (1:N 확장)
        // 컬럼 하나가 여러 테이블에 연결되어 있으면, 여기서 행(Row)이 자동으로 늘어납니다.
        // 예: col(id) -> Table(A), Table(B) ==> 2개의 Row 생성
        MATCH (parent_tbl:Table)-[:HAS_COLUMN]->(col)
        
        // 4. 테이블 별로 다시 그룹핑 (Aggregation)
        // 늘어난 행들을 다시 테이블 이름 기준으로 묶어서 점수를 계산합니다.
        WITH table_hits, 
             parent_tbl.name AS tbl_name, 
             max(c_score) AS max_c_score, 
             count(col) AS c_count, 
             collect(col.name) AS matched_cols
        
        // 5. 최종 반환
        RETURN table_hits, 
               collect({
                   name: tbl_name, 
                   max_c_score: max_c_score, 
                   c_count: c_count, 
                   cols: matched_cols
               }) AS col_hits
        """
        result = tx.run(query, vec=vector, k_tbl=top_k_tables, k_col=top_k_cols)
        
        record = result.single()
        if record is None:
            return {"table_hits": [], "col_hits": []}
        return record

    # [Step 2] Python: 점수 계산 로직
    def _calculate_hybrid_scores(self, raw_data):
        scores = {} # Key: TableName
        
        # 테이블 검색 점수 반영
        for hit in raw_data['table_hits']:
            t_name = hit['tbl']
            scores[t_name] = {
                't_score': hit['score'], 
                'c_max': 0.0, 
                'c_cnt': 0, 
                'columns': []
            }
            
        # 컬럼 검색 점수 반영 (기존에 없으면 추가)
        for hit in raw_data['col_hits']:
            t_name = hit['name']
            if t_name not in scores:
                scores[t_name] = {'t_score': 0.0, 'c_max': 0.0, 'c_cnt': 0, 'columns': []}
            
            scores[t_name]['c_max'] = hit['max_c_score']
            scores[t_name]['c_cnt'] = hit['c_count']
            scores[t_name]['columns'] = hit['cols']

        # 최종 점수 계산
        final_ranking = []
        for t_name, stat in scores.items():
            # 정규화된 Count 점수 (최대 5개까지만 점수 인정)
            norm_cnt = min(stat['c_cnt'], 5) / 5.0 
            
            final_score = (self.W_TABLE * stat['t_score']) + \
                          (self.W_COL_MAX * stat['c_max']) + \
                          (self.W_COL_CNT * norm_cnt)
            
            final_ranking.append({
                'table': t_name, 
                'score': final_score, 
                'relevant_cols': stat['columns']
            })
            
        # 점수 내림차순 정렬
        return sorted(final_ranking, key=lambda x: x['score'], reverse=True)

    # [Step 3] Python: 가지치기 (Adaptive Threshold)
    def _adaptive_pruning(self, ranked_list):
        if not ranked_list: return []
        
        top_score = ranked_list[0]['score']
        cut_off_ratio = 0.9  # 1등 점수의 60% 미만은 버림
        
        selected_tables = []
        for item in ranked_list:
            if item['score'] >= top_score * cut_off_ratio:
                selected_tables.append(item['table'])
            else:
                break # 이미 정렬되어 있으므로 루프 종료
        
        return selected_tables

    # [Step 4] Cypher: 그래프 연결성 확인 및 누락된 연결 고리 찾기
    @staticmethod
    def _ensure_connectivity(tx, table_names):
        if not table_names: return []
        if len(table_names) == 1: return table_names # 하나만 있으면 연결성 체크 불필요
        print(table_names)
        query = """
        // 선택된 테이블 리스트
        UNWIND $names AS name
        MATCH (t:Table {name: name})
        WITH collect(t) AS tables
        
        // 테이블 간의 최단 경로 찾기 (최대 2 hop 정도만 허용하여 너무 멀리 떨어진 건 제외)
        // apoc.algo.dijkstra 등을 쓸 수도 있지만, 여기선 간단히 shortestPath 사용
        MATCH path = shortestPath((t1)-[:HAS_COLUMN*..4]-(t2))
        WHERE t1 IN tables AND t2 IN tables
            AND elementId(t1) < elementId(t2)
        
        // 경로에 포함된 모든 노드(테이블) 추출
        UNWIND nodes(path) AS n
        WITH n
        WHERE 'Table' IN labels(n)

        RETURN DISTINCT n.name AS table_name
        """
        result = tx.run(query, names=table_names)
        return [record["table_name"] for record in result]
#endregion


OPENAI_CLIENT = OpenAI()
RESPONSES_MODEL = os.environ.get("CHATBOT_RESPONSES_MODEL", "gpt-5.2")
MCP_SERVER_URL = "https://wdhdh3.ngrok.io/mcp"
MCP_SERVER_LABEL = "demo-echo"

URI = "bolt:localhost:7687"
AUTH = (os.environ.get("NEO4J_USER", ""), os.environ.get("NEO4J_PASSWORD", ""))
driver = GraphDatabase.driver(URI, auth = AUTH)

retriever = GraphRAGRetriever(driver)

#--- RAG 벡터 스페이스 load ------------------------------------------------------------
persist_dir = Path(os.environ.get("FAISS_INDEX_PATH") or (BASE_DIR / "vectorstores" / "faiss_store"))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    str(persist_dir),
    embeddings,
    allow_dangerous_deserialization=True,
)

docs: list[Document] = vectorstore.docstore._dict.values()
bm25_retriever = BM25Retriever.from_documents(docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
faiss_retriever = vectorstore.as_retriever()
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

PLOT_IMAGE_NAME = "plot"
PLOT_TMP_DIR = "/home"
SQL_EXEC_TRACK_URL = os.environ.get("SQL_EXEC_TRACK_URL", "http://127.0.0.1:1234/ui/sql_exec")
# --- 프롬프트 로더 --------------------------------------------------------------------
def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")

SELECTOR_SYS_PROMPT = load_prompt(PROMPT_DIR / "SELECTOR_SYS_PROMPT.txt")
DECOMPOSER_SYS_PROMPT = load_prompt(PROMPT_DIR / "DECOMPOSER_SYS_PROMPT.txt")
RETRIEVAL_TEMPLATE = load_prompt(PROMPT_DIR / "RETRIEVAL_FORMAT.txt")
CHATBOT_SYS_PROMPT = load_prompt(PROMPT_DIR / "SYS_PROMPT-mod.txt")
PROMODE_SYS_PROMPT = load_prompt(PROMPT_DIR / "PROMODE_SYS_PROMPT.txt")
# --- 유틸리티 -------------------------------------------------------------------------
def evidence_maker(questions: List[str], answers: List[str]) -> str:
    if len(questions) != len(answers):
        raise ValueError("Number of questions and answers does not match.")
    lines: List[str] = []
    for idx, question in enumerate(questions, start=1):
        lines.append(f"Q{idx}. {question}\n")
        lines.append(f"A. {answers[idx - 1]}\n")
    return "".join(lines)

def is_safe_query(sql: str) -> bool:
    """
    Oracle DB용: 데이터 손상 위험이 있는 쿼리를 제외.
    허용: SELECT, WITH, EXPLAIN PLAN
    차단: INSERT, UPDATE, DELETE, MERGE, DROP, ALTER, TRUNCATE, CREATE 등
    """
    sql_upper = sql.strip().upper()

    # 허용된 쿼리만 통과
    safe_prefixes = ("SELECT", "WITH", "EXPLAIN PLAN")
    forbidden_patterns = [
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bDELETE\b",
        r"\bMERGE\b",
        r"\bDROP\b",
        r"\bALTER\b",
        r"\bTRUNCATE\b",
        r"\bCREATE\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bEXECUTE\b",
        r"\bCALL\b",
        r"\bBEGIN\b"
    ]
    if any(re.search(pat, sql_upper) for pat in forbidden_patterns):
        return False
    return True

def table_info_maker(rows, cols) -> str:
    if not rows:
        return "#Table\n[]"
    col_types = []
    first_row = rows[0]
    for value in first_row:
        col_types.append(type(value).__name__)
    result_lines = ["#Table\n["]
    for idx, col in enumerate(cols):
        dtype = col_types[idx] if idx < len(col_types) else "unknown"
        result_lines.append(f" ({col}, type: {dtype})\n")
    result_lines.append("]")
    return "".join(result_lines)

def extract_sql(text: str) -> Optional[str]:
    end_pos = text.rfind("```")
    if end_pos == -1:
        return None
    start_pos = text.rfind("```", 0, end_pos)
    if start_pos == -1:
        return None
    inner_content = text[start_pos + 3 : end_pos]
    first_newline_pos = inner_content.find("\n")
    if first_newline_pos == -1:
        return ""
    content_after_start_line = inner_content[first_newline_pos + 1 :]
    last_newline_pos = content_after_start_line.rfind("\n")
    if last_newline_pos == -1:
        sql = content_after_start_line
    else:
        sql = content_after_start_line[:last_newline_pos]
    return sql.replace(";", "")


def _record_sql_execution(user_id: str) -> None:
    if not user_id:
        return
    endpoint = SQL_EXEC_TRACK_URL.strip()
    if not endpoint:
        return
    payload = {"user_id": user_id}
    try:
        with httpx.Client(timeout=2.0) as client:
            client.post(endpoint, json=payload)
    except httpx.HTTPError as exc:
        print(f"[sql_exec_num] update failed: {exc}", flush=True)


def _safe_parse_json(text: Any) -> Optional[Any]:
    if not text or not isinstance(text, (str, bytes, bytearray)):
        return None
    if isinstance(text, (bytes, bytearray)):
        try:
            text = text.decode("utf-8")
        except Exception:
            return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_response_payload(response) -> Tuple[str, List[Dict[str, Any]]]:
    texts: List[str] = []
    mcp_calls: List[Dict[str, Any]] = []
    for item in getattr(response, "output", []):
        item_type = getattr(item, "type", "")
        if item_type == "message":
            for block in getattr(item, "content", []):
                text = getattr(block, "text", None)
                if text:
                    texts.append(text)
        elif item_type == "mcp_call":
            call_payload: Dict[str, Any] = {
                "id": getattr(item, "id", ""),
                "name": getattr(item, "name", ""),
                "arguments": getattr(item, "arguments", ""),
                "output": getattr(item, "output", None),
                "error": getattr(item, "error", None),
            }
            parsed_output = _safe_parse_json(call_payload["output"])
            if isinstance(parsed_output, dict):
                call_payload["output_data"] = parsed_output
            mcp_calls.append(call_payload)
    return "\n".join(texts), mcp_calls

def _ensure_plot_container(client: docker.DockerClient, user_id: str):
    try:
        container = client.containers.get(user_id)
        container.reload()
        if container.status != "running":
            container.start()
    except docker.errors.NotFound:
        container = client.containers.run(
            PLOT_IMAGE_NAME,
            command="sleep infinity",
            detach=True,
            name=user_id,
        )
    return container

def _allocate_csv_path(container) -> Tuple[str, str]:
    list_command = f"ls {PLOT_TMP_DIR}/output*.csv 2>/dev/null || true"
    _, output = container.exec_run(["/bin/sh", "-c", list_command])
    decoded = output.decode("utf-8") if isinstance(output, (bytes, bytearray)) else str(output)
    indices = [int(match) for match in re.findall(r"output(\d+)\.csv", decoded)]
    next_index = max(indices) + 1 if indices else 1
    filename = f"output{next_index}.csv"
    return filename, f"{PLOT_TMP_DIR}/{filename}"

def run_sql(
    sql: str,
    db_name: str,
    user_id: str
) -> Dict[str, Any]:
    """
    sql 가져와서 실행하고 컨테이너 생성 후 그 안에 저장하는 함수.
    """
    user = ORACLE_JSON[db_name]["ORACLE_USER"]
    host = ORACLE_JSON[db_name]["ORACLE_HOST"]
    port = ORACLE_JSON[db_name]["ORACLE_PORT"]
    service_name = ORACLE_JSON[db_name]["ORACLE_SERVICE_NAME"]
    password = ORACLE_JSON[db_name]["ORACLE_PASSWORD"]
    if not is_safe_query(sql):
        print("Not Safe!")
    oracledb.init_oracle_client(lib_dir=os.environ.get("ORACLE_CLIENT_LIB_DIR") or None)
    try: conn = oracledb.connect(user=user, password = password, dsn=f"{host}:{port}/{service_name}")
    except:
        print("error")
        exit
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    #csv 생성
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(cols)
    writer.writerows(rows)
    csv_text = buffer.getvalue()
    #container 생성 및 파일 경로 결정
    client = docker.from_env()
    container = _ensure_plot_container(client, user_id)
    dest_filename, dest_path_in_container = _allocate_csv_path(container)
    pw_tarstream = io.BytesIO()
    encoded_csv = csv_text.encode("utf-8")
    with tarfile.open(fileobj=pw_tarstream, mode="w") as tar:
        tarinfo = tarfile.TarInfo(name=dest_filename)
        tarinfo.size = len(encoded_csv)
        tar.addfile(tarinfo, io.BytesIO(encoded_csv))
    pw_tarstream.seek(0)
    dest_dir = os.path.dirname(dest_path_in_container)
    container.put_archive(path=dest_dir, data=pw_tarstream.getvalue())
    verify_command = f"cat {dest_path_in_container}"
    exit_code, output = container.exec_run(verify_command)
    if exit_code == 0:
        print(output.decode().strip())
        print(f"CSV saved to {dest_path_in_container}")
    else:
        print("Error verifying file:")
        print(output.decode().strip())

    table_info = table_info_maker(rows, cols)
    serializable_rows = [["" if cell is None else str(cell) for cell in row] for row in rows]

    _record_sql_execution(user_id)

    return {
        "table_info": table_info,
        "csv": csv_text,
        "headers": [str(col) for col in cols],
        "rows": serializable_rows,
        "csv_path": dest_path_in_container,
    }


def format_docs(docs: List[Document]) -> str:
    """
    Document 객체 리스트의 page_content를 
    하나의 문자열로 합칩니다.
    """
    return "\n\n".join(doc.page_content for doc in docs)
# --- LLM 세팅 -------------------------------------------------------------------------
local_llm = ChatOpenAI(model="models/Qwen3-Coder-30B-A3B-Instruct", base_url = "http://127.0.0.1:8000/v1", openai_api_key="dummy", reasoning = {"effort": "medium"})
"""
high_llm = ChatOpenAI(model="gpt-5-mini", temperature=0, reasoning = {"effort": "high"})
medium_llm = ChatOpenAI(model="gpt-5-mini", temperature=0, reasoning = {"effort": "medium"})
low_llm = ChatOpenAI(model="gpt-5-mini", temperature=0, reasoning= {"effort": "low"})

"""
class PromodeOutput(BaseModel):
    is_enough: bool = Field(...)
    llm_question: List[str] = Field(...)

class TableSelection(BaseModel):
    selection: Dict[str, Union[str, List[str]]] = Field(
        description=(
            "A dictionary where keys are table names. Values can be 'keep_all', "
            "'drop_all', or a list of relevant column names."
        )
    )

class BatchTranslation(BaseModel):
    translated_text: List[str] = Field(...)
"""
def clean_and_json(output: str):
    result = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    return json.loads(result)
"""
selector_prompt = ChatPromptTemplate.from_template(SELECTOR_SYS_PROMPT)
selector_parser = JsonOutputParser(pydantic_object=TableSelection)
selector_chain = selector_prompt | local_llm | selector_parser

retriever_chain = ensemble_retriever | RunnableLambda(format_docs)

decomposer_prompt = ChatPromptTemplate.from_template(DECOMPOSER_SYS_PROMPT)
decomposer_chain = decomposer_prompt | local_llm | StrOutputParser()

promode_prompt = ChatPromptTemplate.from_template(PROMODE_SYS_PROMPT)
promode_parser = JsonOutputParser(pydantic_object=PromodeOutput)
promode_chain = promode_prompt | local_llm | promode_parser
# --- 상태 정의 ------------------------------------------------------------------------
class AnalysisState(TypedDict, total=False):
    user_prompt: str
    db_id: str
    desc_str: str
    fk_str: str
    evidence: str
    formal_question: str
    context: str
    promode_questions: List[str]


class DirectState(TypedDict, total=False):
    user_prompt: str
    db_id: str
    desc_str: str
    fk_str: str
    evidence: str
    selector_output: Dict[str, Union[str, List[str]]]
    processed_desc_str: str
    final_output: str
    context: str

def promode_node(state: AnalysisState) -> AnalysisState:
    start = time.time()
    result = promode_chain.invoke(
        {
            "desc_str": state.get("desc_str", ""),
            "query": state["user_prompt"],
            "evidence": "",
            "context": ""
        }
    )
    elapsed = time.time() -start
    print(f"Promode took {elapsed:.4f} seconds")
    state["promode_questions"] = result.get("llm_question", [])
    return state

def direct_retrieval_node(state: DirectState) -> DirectState:
    start = time.time()
    query = state.get("user_prompt", "")
    context = retriever_chain.invoke(query)
    elapsed = time.time() - start
    print(f"Context retrieval took {elapsed:.4f} seconds; length={len(context)}")
    print(context)
    state["context"] = context
    return state

def decomposer_node_direct(state: DirectState) -> DirectState:
    start = time.time()
    final_output = decomposer_chain.invoke(
        {
            "db_id": state["db_id"],
            "query": state.get("formal_question", state["user_prompt"]),
            "evidence": state.get("evidence", ""),
            "desc_str": state["desc_str"],
            "fk_str": state["fk_str"],
            "context": state["context"]
        }
    )
    #print(state.get("processed_desc_str"))
    elapsed = time.time() - start
    #print(final_output)
    print(f"Decomposer took {elapsed:.4f} seconds")
    state["final_output"] = final_output
    return state
# --- 그래프 구성 ----------------------------------------------------------------------
def build_direct_graph() -> StateGraph:
    graph: StateGraph = StateGraph(DirectState)
    graph.add_node("decomposer", decomposer_node_direct)
    graph.add_node("retrieval", direct_retrieval_node)

    graph.add_edge(START, "decomposer")
    graph.add_edge("decomposer", END)
    return graph

def build_retreival_graph() -> StateGraph:
    graph: StateGraph = StateGraph(DirectState)
    graph.add_node("retrieval", direct_retrieval_node)

    graph.add_edge(START, "retrieval")
    graph.add_edge("retrieval", END)
    return graph

def build_promode_graph() -> StateGraph:
    graph: StateGraph = StateGraph(AnalysisState)
    graph.add_node("promode", promode_node)

    graph.add_edge(START, "promode")
    graph.add_edge("promode", END)
    return graph

DIRECT_GRAPH = build_direct_graph().compile()
PROMODE_GRAPH = build_promode_graph().compile()
RETRIEVAL_GRAPH = build_retreival_graph().compile()
# --- 공개 API -------------------------------------------------------------------------
class DirectResult(TypedDict, total=False):
    is_relevant: bool
    formal_question: str
    selector_output: Dict[str, Union[str, List[str]]]
    processed_desc_str: str
    final_output: str


class FinalResult(TypedDict, total=False):
    final_output: str

SCHEMA_CACHE: Dict[Tuple[Path, str], Tuple[str, str]] = {}

def get_schema(db_id: str, tables_json_path: Path) -> Tuple[str, str]:
    key = (tables_json_path, db_id)
    if key not in SCHEMA_CACHE:
        SCHEMA_CACHE[key] = generate_schema_strings(tables_json_path, db_id)
    return SCHEMA_CACHE[key]

def run_promode(
    question: str,
    db_id: str
) -> List[str]:
    import ipdb; ipdb.set_trace()
    desc_str = retriever.get_relevant_schema(question)
    fk_str = ""
    state: AnalysisState = {
        "user_prompt": question,
        "db_id": db_id,
        "desc_str": desc_str,
        "fk_str": fk_str
    }
    result = PROMODE_GRAPH.invoke(state)
    return result.get("promode_questions", [])

def run_direct(
    question: str,
    evidence: str,
    db_id: str
) -> str:
    #DirectResult:
    state: DirectState = {
        "user_prompt": question
    }
    result = RETRIEVAL_GRAPH.invoke(state)
    desc_str = retriever.get_relevant_schema(question, result["context"])
    print(desc_str)
    fk_str = ""
    state: DirectState = {
        "user_prompt": question,
        "db_id": db_id,
        "desc_str": desc_str,
        "fk_str": fk_str,
        "evidence": evidence,
        "context": result["context"]
    }
    result = DIRECT_GRAPH.invoke(state)
    return extract_sql(result.get("final_output"))

def run_chat_turn(
    user_prompt: str,
    chat_history: List[Dict[str, str]],
    user_id: str
) -> Dict[str, Any]:
    chat_history.append({"role": "user", "content": user_prompt})
    sanitized_history: List[Dict[str, str]] = []
    for entry in chat_history:
        if entry.get("type", "") != "":
            # Responses API가 허용하지 않는 사용처 전용 로그는 제외한다.
            continue
        role = entry.get("role")
        if role is None:
            continue
        content = entry.get("content")
        sanitized_entry: Dict[str, str] = {
            "role": str(role),
            "content": str(content) if content is not None else "",
        }
        sanitized_history.append(sanitized_entry)
    for chat in sanitized_history:
        if chat["role"] != "system":
            print(chat)
    tools_payload = _build_tools_payload()
    start = time.time()
    response = OPENAI_CLIENT.responses.create(
        model=RESPONSES_MODEL,
        input=sanitized_history,
        tools=tools_payload,
        reasoning={"effort": "medium"},
        tool_choice="auto" if tools_payload else "none",
    )
    elapsed = time.time() - start
    print(f"API took {elapsed:.4f} seconds")
    assistant_reply, mcp_calls = _extract_response_payload(response)
    assistant_reply = assistant_reply.strip()

    pending_job: Optional[Dict[str, Any]] = None
    latest_job_data: Optional[Dict[str, Any]] = None
    for call in mcp_calls:
        if call.get("name") in {"text_to_sql", "check_text_to_sql"}:
            payload = call.get("output_data")
            if isinstance(payload, dict):
                latest_job_data = payload

    if latest_job_data:
        status = str(latest_job_data.get("status", "")).lower()
        if not status and latest_job_data.get("job_id"):
            status = "queued"
        if status in {"queued", "running"}:
            pending_job = {
                "job_id": latest_job_data.get("job_id"),
                "status": status,
                "suggested_poll_interval": latest_job_data.get("suggested_poll_interval", latest_job_data.get("poll_interval", 5)),
                "question": latest_job_data.get("question"),
                "user_id": latest_job_data.get("user_id"),
            }
            assistant_reply = ""
        elif status in {"failed", "not_found"} and not assistant_reply:
            error_msg = latest_job_data.get("error") or "원인을 확인하지 못했습니다."
            assistant_reply = f"SQL 생성에 실패했습니다. 사유: {error_msg}"
        elif status == "succeeded" and not assistant_reply:
            result_sql = latest_job_data.get("result")
            if result_sql:
                trimmed_sql = str(result_sql).strip()
                assistant_reply = (
                    "**SQL**\n``` sql\n"
                    f"{trimmed_sql}\n"
                    "```\n``` description\n"
                    "사용자 요청에 따라 생성된 SQL입니다.\n"
                    "```"
                )

    if assistant_reply:
        chat_history.append({"role": "assistant", "content": assistant_reply})
        print(chat_history)

    if assistant_reply.startswith("**PLOT**"):
        client = docker.from_env()
        container = _ensure_plot_container(client, user_id)
        try: stream, stat = container.get_archive(f"{PLOT_TMP_DIR}/output.png")
        except: exit()
        file_like = io.BytesIO(b"".join(stream))
        with tarfile.open(fileobj=file_like) as tar:
            member = tar.getmembers()[0]
            file_data = tar.extractfile(member).read()
        b64_encoded = base64.b64encode(file_data).decode("utf-8")
        b64_encoded = f"data:image/png;base64,{b64_encoded}"
        return {
            "response": assistant_reply,
            "chat_history": chat_history,
            "image": b64_encoded
        }

    return {
        "response": assistant_reply,
        "chat_history": chat_history,
        "pending_job": pending_job,
    }

def _build_tools_payload() -> List[Dict[str, Any]]:
    if not MCP_SERVER_URL:
        return []
    return [
        {
            "type": "mcp",
            "server_label": MCP_SERVER_LABEL,
            "server_url": MCP_SERVER_URL,
            "require_approval": "never",
        }
    ]
