from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import CountVectorizer

from agent_pipeline.session_state import SessionState


def make_tools(session: SessionState) -> List[Any]:
    """Factory function to build the document-term-matrix tools bound to the session."""

    @tool
    def build_dtm_tool(
        ngram_min: int = 1,
        ngram_max: int = 1,
        stop_words: str = None,
        max_features: int = None,
    ) -> Dict[str, Any]:
        """Build the document-term matrix of the working DataFrame's text with
        scikit-learn's CountVectorizer and store it in the session (feature_matrix,
        feature_names, labels), for every later tool to use. Reports the matrix's sparsity.
        The fitted vectorizer is kept in session.artifacts["count_vectorizer"].

        Args:
            ngram_min: smallest n-gram length, 1 for single words.
            ngram_max: largest n-gram length, 1 for single words only; must be >= ngram_min.
            stop_words: pass "english" to drop English stop words, or leave empty to keep every word.
            max_features: keep only the most frequent terms, or leave empty to keep all of them.

        Returns:
            Dict[str, Any]: result_id, n_documents, n_terms, non_zero, total_elements and sparsity_pct.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please call load_dataset_tool first."}
        if ngram_min < 1 or ngram_max < ngram_min:
            return {"error": "ngram_min must be >= 1 and ngram_max must be >= ngram_min."}
        if stop_words not in (None, "english"):
            return {"error": "stop_words must be 'english' or left empty."}

        df = session.dataframe
        vectorizer = CountVectorizer(
            ngram_range=(ngram_min, ngram_max),
            stop_words=stop_words,
            max_features=max_features,
        )
        matrix = vectorizer.fit_transform(df["text"].fillna("").astype(str))

        session.feature_matrix = matrix
        session.feature_names = vectorizer.get_feature_names_out().tolist()
        session.set_labels(df["category_name"].astype(str).values)
        session.artifacts["count_vectorizer"] = vectorizer

        total_elements = matrix.shape[0] * matrix.shape[1]
        non_zero = int(matrix.nnz)
        sparsity_pct = 100 * (1 - non_zero / total_elements) if total_elements else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": int(matrix.shape[0]),
            "n_terms": int(matrix.shape[1]),
            "non_zero": non_zero,
            "total_elements": int(total_elements),
            "sparsity_pct": round(float(sparsity_pct), 4),
        }
        session.store_result(
            "dtm",
            {"ngram_min": ngram_min, "ngram_max": ngram_max, "stop_words": stop_words, "max_features": max_features},
            summary,
        )
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Add up each term's count over all documents of the global document-term matrix
        and list the most frequent terms. Needs build_dtm_tool first. It only reports the
        totals and does not filter anything.

        Args:
            top_n: how many of the most frequent terms to list in the summary.

        Returns:
            Dict[str, Any]: result_id, n_terms, total_count and top_terms (term and frequency,
            most frequent first). The full ranking is stored for visualize_result_tool.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix yet. Please call build_dtm_tool first."}
        if top_n < 1:
            return {"error": "top_n must be at least 1."}

        totals = session.feature_matrix.sum(axis=0).A1
        report = (
            pd.DataFrame({"term": session.feature_names, "frequency": totals})
            .sort_values("frequency", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "n_terms": int(len(report)),
            "total_count": int(totals.sum()),
            "top_terms": [
                {"term": str(row.term), "frequency": int(row.frequency)} for row in report.head(top_n).itertuples()
            ],
        }
        session.store_result("term_frequency", {"top_n": top_n}, summary, full_report=report)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Draw a heatmap of the raw counts of a slice of the global document-term matrix:
        the first n_documents rows and first n_terms columns, in their existing order, with
        no sorting or filtering. Needs build_dtm_tool first. Asking for more than exists
        simply shows everything.

        Args:
            n_terms: how many terms (columns) to show, taken in vocabulary order.
            n_documents: how many documents (rows) to show, taken in row order.

        Returns:
            Dict[str, Any]: result_id, terms, documents (labels) and matrix (one list per document).
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix yet. Please call build_dtm_tool first."}
        if n_terms < 1 or n_documents < 1:
            return {"error": "n_terms and n_documents must be at least 1."}

        block = session.feature_matrix[0:n_documents, 0:n_terms].toarray()
        terms = [str(t) for t in session.feature_names[0:n_terms]]
        index = session.dataframe.index if session.dataframe is not None else None
        if index is not None and len(index) == session.feature_matrix.shape[0]:
            documents = [f"doc_{i}" for i in list(index)[0:n_documents]]
        else:
            documents = [f"doc_{i}" for i in range(block.shape[0])]

        fig = Figure(figsize=(12, 8))
        ax = fig.subplots()
        sns.heatmap(
            pd.DataFrame(block, columns=terms, index=documents),
            cmap="YlGnBu",
            vmin=0,
            vmax=max(int(block.max()), 1),
            annot=True,
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"Document-Term Matrix Heatmap ({block.shape[1]} terms, {block.shape[0]} documents)")
        fig.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "terms": terms,
            "documents": documents,
            "matrix": block.tolist(),
        }
        session.store_result("dtm_heatmap", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
