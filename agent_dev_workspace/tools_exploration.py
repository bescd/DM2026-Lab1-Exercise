from typing import Any, Dict, List

import numpy as np
import seaborn as sns
from langchain_core.tools import tool
from matplotlib.figure import Figure
from sklearn.metrics.pairwise import cosine_similarity

from agent_pipeline.session_state import SessionState


def make_tools(session: SessionState) -> List[Any]:
    """Factory function to build the exploration tools bound to the session."""

    @tool
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> Dict[str, Any]:
        """Cosine similarity between two documents, from their raw word counts, using the same
        CountVectorizer that built the document-term matrix. 1 means the same word proportions,
        0 means no word in common (a document with no words gives exactly 0). Needs
        build_dtm_tool first.

        Args:
            doc_index_1: position of the first document, 0 being the first row of the DataFrame.
            doc_index_2: position of the second document, 0 being the first row of the DataFrame.

        Returns:
            Dict[str, Any]: result_id, doc_index_1, doc_index_2, their category_name and cosine_similarity.
        """
        vectorizer = session.artifacts.get("count_vectorizer")
        if session.feature_matrix is None or vectorizer is None or session.dataframe is None:
            return {"error": "No document-term matrix yet. Please call build_dtm_tool first."}

        df = session.dataframe
        n_documents = len(df)
        for position in (doc_index_1, doc_index_2):
            if position < 0 or position >= n_documents:
                return {"error": f"Document position {position} is out of range (0 to {n_documents - 1})."}

        texts = [str(df["text"].fillna("").iloc[p]) for p in (doc_index_1, doc_index_2)]
        vectors = vectorizer.transform(texts)
        similarity = float(cosine_similarity(vectors[0], vectors[1])[0, 0])
        if not np.isfinite(similarity):
            similarity = 0.0

        result_id = session.next_result_id("cosine")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "category_name_1": str(df.iloc[doc_index_1]["category_name"]),
            "category_name_2": str(df.iloc[doc_index_2]["category_name"]),
            "cosine_similarity": round(similarity, 4),
        }
        session.store_result("cosine", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Pearson correlation between the top_n terms with the highest variance of the global
        document-term matrix, drawn as an annotated heatmap. Shows which frequent terms move
        together. Needs build_dtm_tool first. If the vocabulary has fewer than top_n terms, all
        of them are used.

        Args:
            top_n: how many of the highest-variance terms to correlate.

        Returns:
            Dict[str, Any]: result_id, terms (highest variance first) and correlation_matrix
            (rounded to 4 decimals, same order on both axes).
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix yet. Please call build_dtm_tool first."}
        if top_n < 2:
            return {"error": "top_n must be at least 2."}

        dense = session.feature_matrix.toarray()
        variances = dense.var(axis=0)
        top_indices = np.argsort(variances)[-top_n:][::-1]
        terms = [str(session.feature_names[i]) for i in top_indices]
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.corrcoef(dense[:, top_indices].T)

        fig = Figure(figsize=(15, 12))
        ax = fig.subplots()
        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt=".2f",
            xticklabels=terms,
            yticklabels=terms,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Heatmap (Top {len(terms)} Features by Variance)", fontsize=14)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "terms": terms,
            # A constant column has no defined correlation: reported as null, never as NaN.
            "correlation_matrix": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in correlation],
        }
        session.store_result("feature_correlation", {"top_n": top_n}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
