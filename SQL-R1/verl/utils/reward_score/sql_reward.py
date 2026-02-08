
import re
import sqlite3
import sqlparse
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from collections import defaultdict
import timeout_decorator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Breakdown of reward components for analysis"""
    execution_correct: float = 0.0
    syntax_valid: float = 0.0
    schema_aligned: float = 0.0
    keyword_match: float = 0.0
    structure_similarity: float = 0.0
    total: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'execution_correct': self.execution_correct,
            'syntax_valid': self.syntax_valid,
            'schema_aligned': self.schema_aligned,
            'keyword_match': self.keyword_match,
            'structure_similarity': self.structure_similarity,
            'total': self.total
        }


class SQLRewardFunction:
    """
    Comprehensive reward function for Text-to-SQL RL training
    
    Reward Structure:
    - Execution Match: 1.0 (if results match gold query)
    - Syntax Valid: 0.1 (if SQL is parseable)
    - Schema Aligned: 0.2 (if uses correct tables/columns)
    - Keyword Match: 0.1 (if uses similar SQL operations)
    - Structure Similarity: 0.1 (if query structure is similar)
    
    Total possible: 1.5 (to encourage exploration beyond just correctness)
    """
    
    def __init__(
        self,
        db_path: str,
        weights: Optional[Dict[str, float]] = None,
        timeout_seconds: int = 5
    ):
        self.db_path = db_path
        self.timeout_seconds = timeout_seconds
        
        # Default weights for reward components
        self.weights = weights or {
            'execution': 1.0,
            'syntax': 0.1,
            'schema': 0.2,
            'keywords': 0.1,
            'structure': 0.1
        }
        
        # Statistics tracking
        self.stats = defaultdict(int)
        
    def execute_query(self, query: str) -> Tuple[Optional[List[Tuple]], Optional[str]]:
        """
        Safely execute SQL query
        
        Returns:
            (results, error_message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Use timeout_decorator dynamically
            try:
                timed_execute = timeout_decorator.timeout(self.timeout_seconds, use_signals=False)(self._execute_query_raw)
                results = timed_execute(query, conn)
            except (timeout_decorator.TimeoutError, TimeoutError):
                return None, f"Query timeout (>{self.timeout_seconds}s)"
            except Exception as e:
                # Fallback for systems where timeout_decorator/multiprocessing fails (like some Windows environments)
                if "handle is invalid" in str(e) or "PicklingError" in str(e):
                    results = self._execute_query_raw(query, conn)
                else:
                    raise e
            
            conn.close()
            return results, None
        except sqlite3.Error as e:
            return None, f"SQL Error: {str(e)}"
        except Exception as e:
            return None, f"Execution error: {str(e)}"

    def _execute_query_raw(self, query: str, conn: sqlite3.Connection) -> List[Tuple]:
        """Raw execution without timeout (wrapped by caller)"""
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()

    def check_syntax_valid(self, query: str) -> bool:
        """Check if SQL query is syntactically valid"""
        if not query or not isinstance(query, str):
            return False
        try:
            parsed = sqlparse.parse(query)
            if not parsed or len(parsed) == 0:
                return False
            
            # Check if it's actually a SQL statement
            stmt = parsed[0]
            stmt_type = stmt.get_type()
            return stmt_type in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']
        except Exception:
            return False
    
    def extract_schema_elements(self, query: str) -> Tuple[set, set]:
        """
        Extract table and column references from query using sqlparse
        
        Returns:
            (tables, columns)
        """
        tables = set()
        columns = set()
        
        try:
            parsed = sqlparse.parse(query)
            if not parsed:
                return tables, columns
            
            stmt = parsed[0]
            
            # Use a more robust token walker
            def walk_tokens(tokens):
                in_from = False
                for token in tokens:
                    # Identify clauses that change context
                    if token.is_keyword:
                        val = token.value.upper()
                        if val in ('FROM', 'JOIN', 'INTO', 'UPDATE'):
                            in_from = True
                        elif val in ('SELECT', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'SET', 'VALUES'):
                            in_from = False
                    
                    # Handle identifiers
                    if isinstance(token, sqlparse.sql.Identifier):
                        name = token.get_real_name()
                        if name:
                            if in_from:
                                tables.add(name)
                            else:
                                columns.add(name)
                            
                            # Check for parent (table.col)
                            parent = token.get_parent_name()
                            if parent:
                                tables.add(parent)
                                
                    elif isinstance(token, sqlparse.sql.IdentifierList):
                        for identifier in token.get_identifiers():
                            if isinstance(identifier, sqlparse.sql.Identifier):
                                name = identifier.get_real_name()
                                if name:
                                    if in_from:
                                        tables.add(name)
                                    else:
                                        columns.add(name)
                                    parent = identifier.get_parent_name()
                                    if parent:
                                        tables.add(parent)
                    
                    # Recurse into groups (Parenthesis, etc)
                    if token.is_group:
                        walk_tokens(token.tokens)
            
            walk_tokens(stmt.tokens)
                                    
        except Exception as e:
            logger.debug(f"sqlparse error: {e}. Falling back to regex.")
            # Fallback to regex if parsing fails structurally
            return self._extract_schema_regex(query)
            
        return tables, columns

    def _extract_schema_regex(self, query: str) -> Tuple[set, set]:
        """Fallback regex extraction"""
        query_lower = query.lower()
        tables = set()
        # clauses that usually precede table names
        table_keywords = ['from', 'join', 'update', 'into']
        for kw in table_keywords:
            tables.update(re.findall(rf'{kw}\s+([a-zA-Z_][a-zA-Z0-9_]*)', query_lower))
        
        # Simple column extraction: any identifier that isn't a keyword or table
        all_ids = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', query)
        
        # We'll use sqlparse's keyword list for better filtering if available
        # or a standard set if not
        sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'JOIN', 'ON', 'AS', 'IN', 'IS', 'NOT', 'NULL',
            'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DISTINCT', 'ORDER', 'BY', 'GROUP', 'HAVING', 
            'LIMIT', 'LIKE', 'BETWEEN', 'INTO', 'UPDATE', 'VALUES', 'SET', 'DELETE', 'CREATE', 'TABLE'
        }
        
        columns = {id_v for id_v in all_ids if id_v.upper() not in sql_keywords and id_v not in tables}
        
        return tables, columns
    
    def compute_schema_alignment(self, pred_query: str, gold_query: str) -> float:
        """
        Compute schema alignment score based on table/column overlap
        
        Returns:
            Score in [0, 1]
        """
        pred_tables, pred_cols = self.extract_schema_elements(pred_query)
        gold_tables, gold_cols = self.extract_schema_elements(gold_query)
        
        if not gold_tables and not gold_cols:
            return 1.0  # Edge case
        
        # Compute Jaccard similarity for tables and columns
        table_score = 0.0
        if gold_tables:
            table_score = len(pred_tables & gold_tables) / len(gold_tables | pred_tables)
        
        col_score = 0.0
        if gold_cols:
            col_score = len(pred_cols & gold_cols) / len(gold_cols | pred_cols)
        
        # Weighted average (tables are more important)
        return 0.6 * table_score + 0.4 * col_score
    
    def extract_keywords(self, query: str) -> set:
        """Extract SQL keywords using sqlparse token types"""
        keywords = set()
        try:
            parsed = sqlparse.parse(query)
            if not parsed:
                return keywords
            
            # Flatten tokens to easily iterate
            for token in parsed[0].flatten():
                if token.ttype in [sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DML]:
                    keywords.add(token.value.upper())
                # Capturing core SQL concepts (SELECT, FROM, WHERE, etc)
                # DML covers SELECT, INSERT, UPDATE, DELETE
                # Keyword covers FROM, WHERE, JOIN, ORDER BY, etc.
        except Exception:
            # Fallback to simple splitting if parsing fails
            return set(query.upper().split())
        
        return keywords
    
    def compute_keyword_similarity(self, pred_query: str, gold_query: str) -> float:
        """
        Compute keyword overlap score
        
        Returns:
            Score in [0, 1]
        """
        pred_keywords = self.extract_keywords(pred_query)
        gold_keywords = self.extract_keywords(gold_query)
        
        if not gold_keywords:
            return 1.0
        
        return len(pred_keywords & gold_keywords) / len(gold_keywords | pred_keywords)
    
    # Standard SQL clauses for structural similarity
    STRUCTURE_CLAUSES = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT']

    def compute_structure_similarity(self, pred_query: str, gold_query: str) -> float:
        """
        Compute structural similarity based on clause presence and order
        
        Returns:
            Score in [0, 1]
        """
        def get_clause_structure(query: str) -> List[str]:
            """Extract clause types in order"""
            query_upper = query.upper()
            structure = []
            
            for clause in self.STRUCTURE_CLAUSES:
                if clause in query_upper:
                    structure.append(clause)
            
            return structure
        
        pred_structure = get_clause_structure(pred_query)
        gold_structure = get_clause_structure(gold_query)
        
        if not gold_structure:
            return 1.0
        
        # Compute sequence similarity (simple version)
        matches = sum(1 for p, g in zip(pred_structure, gold_structure) if p == g)
        return matches / max(len(pred_structure), len(gold_structure))
    
    def compute_score(
        self, 
        pred_query: str, 
        gold_query: str,
    ) -> float:
        """
        Compute comprehensive reward for predicted SQL query
        
        Args:
            pred_query: Predicted SQL query
            gold_query: Ground truth SQL query
            
        Returns:
            Total reward
        """
        components = RewardComponents()
        
        # 0. Basic validation
        if not pred_query or not gold_query:
            return 0.0

        # 1. Execution Correctness (Primary Reward)
        pred_results, pred_error = self.execute_query(pred_query)
        gold_results, gold_error = self.execute_query(gold_query)
        
        if pred_error is None and gold_error is None and pred_results is not None:
            # Sort results for comparison (order shouldn't matter unless ORDER BY is present)
            # But here we do a simple check. Robust comparison is in exec_eval.py if needed.
            # For RL reward, we use a slightly simpler check or rely on exec_eval if integrated.
            pred_sorted = sorted([tuple(row) for row in pred_results])
            gold_sorted = sorted([tuple(row) for row in gold_results])
            
            if pred_sorted == gold_sorted:
                components.execution_correct = self.weights['execution']
                self.stats['execution_correct'] += 1
            else:
                # Partial credit for same number of rows (if they have rows)
                if len(pred_results) == len(gold_results) and len(gold_results) > 0:
                    components.execution_correct = 0.3 * self.weights['execution']
                    self.stats['partial_execution'] += 1
        
        # 2. Syntax Validity
        if self.check_syntax_valid(pred_query):
            components.syntax_valid = self.weights['syntax']
            self.stats['syntax_valid'] += 1
        
        # 3. Schema Alignment (only if syntax is valid)
        if components.syntax_valid > 0:
            schema_score = self.compute_schema_alignment(pred_query, gold_query)
            components.schema_aligned = schema_score * self.weights['schema']
        
        # 4. Keyword Similarity
        keyword_score = self.compute_keyword_similarity(pred_query, gold_query)
        components.keyword_match = keyword_score * self.weights['keywords']
        
        # 5. Structure Similarity
        structure_score = self.compute_structure_similarity(pred_query, gold_query)
        components.structure_similarity = structure_score * self.weights['structure']
        
        # Total reward
        components.total = (
            components.execution_correct +
            components.syntax_valid +
            components.schema_aligned +
            components.keyword_match +
            components.structure_similarity
        )
        
        self.stats['total_queries'] += 1
        return components.total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reward computation statistics"""
        total = self.stats['total_queries']
        if total == 0:
            return {}
        
        return {
            'total_queries': total,
            'execution_correct': self.stats['execution_correct'],
            'execution_correct_rate': self.stats['execution_correct'] / total if total > 0 else 0,
            'partial_execution': self.stats['partial_execution'],
            'syntax_valid': self.stats['syntax_valid'],
            'syntax_valid_rate': self.stats['syntax_valid'] / total if total > 0 else 0
        }

# ... (existing imports)

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    Copied from synsql.py to ensure consistent extraction.
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        # If no header found, assume the whole string is the response (or already processed)
        processed_str = solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def parse_sql_from_answer(answer_text: str) -> Optional[str]:
    """Parses SQL from the model's answer text."""
    sql_pattern = r'```sql(.*?)```'
    matches = list(re.finditer(sql_pattern, answer_text, re.DOTALL))
    
    if not matches:
        # Fallback: try to find SELECT ...
        # But for now, strictly follow format
        return None
    
    return matches[-1].group(1).strip()

# Global instance for easier integration
_REWARD_FN = None

def compute_score(solution_str, ground_truth, db_path='data/database.db'):
    global _REWARD_FN
    
    # Dynamic DB Path Resolution
    # Some datasets (like Spider/SynSQL) provide the DB info in ground_truth
    final_db_path = db_path
    if isinstance(ground_truth, dict):
        db_id = ground_truth.get('db_id')
        db_base_path = ground_truth.get('db_base_path')
        if db_id and db_base_path:
            # Construct path: base/db_id/db_id.sqlite
            final_db_path = os.path.join(db_base_path, db_id, f"{db_id}.sqlite")
    
    if _REWARD_FN is None or _REWARD_FN.db_path != final_db_path:
        _REWARD_FN = SQLRewardFunction(db_path=final_db_path)
    
    # 1. Extract Pred SQL
    answer_text, processed_str = extract_solution(solution_str)
    pred_sql = None
    if answer_text:
        pred_sql = parse_sql_from_answer(answer_text)
    
    # Fallback if specific tags are missing but query looks like SQL (e.g. raw output)
    if not pred_sql and not answer_text:
         # simple heuristic if model just outputted SQL
         if "SELECT" in solution_str.upper():
             pred_sql = solution_str
             # Try to strip markdown if present
             matches = list(re.finditer(r'```sql(.*?)```', solution_str, re.DOTALL))
             if matches:
                 pred_sql = matches[-1].group(1).strip()
    
    if not pred_sql:
        # No SQL found -> 0 reward
        return 0.0

    # 2. Extract Gold SQL
    gold_sql = ""
    if isinstance(ground_truth, dict):
        gold_sql = ground_truth.get('sql', '')
    else:
        gold_sql = str(ground_truth)
    
    # Normalize Gold SQL
    gold_sql = re.sub(r'\s+', ' ', gold_sql).strip()
    pred_sql = re.sub(r'\s+', ' ', pred_sql).strip()

    return _REWARD_FN.compute_score(pred_sql, gold_sql)
