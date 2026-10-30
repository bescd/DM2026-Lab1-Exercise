from typing import Any, Dict, List

from langchain_core.tools import tool

from agent_pipeline.session_state import SessionState
from helpers.feature_filtering import pearson_filter, spearman_filter, variance_filter


def make_tools(session: SessionState) -> List[Any]:
    """Factory function to build the feature-filtering tools bound to the session."""

    def _binary_target(target_class: str):
        """One-vs-rest 0/1 vector for target_class, or an error dict if it can't be built."""
        if session.feature_matrix is None:
            return None, {"error": "No document-term matrix yet. Please call build_dtm_tool first."}
        if target_class not in session.categories:
            return None, {
                "error": f"Unknown target_class '{target_class}'.",
                "available_categories": list(session.categories),
            }
        return (session.labels == target_class).astype(int), None

    @tool
    def variance_filter_tool(threshold: float = 0.0, top_n: int = 20) -> Dict[str, Any]:
        """Flag near-constant terms of the global document-term matrix by their variance
        across documents (population variance, like scikit-learn's VarianceThreshold). A term
        is kept if its variance is strictly above the threshold. Only reports which terms
        would be kept or removed; the matrix itself is not modified. Needs build_dtm_tool first.

        Args:
            threshold: terms with variance at or below this value are flagged for removal.
            top_n: how many terms to list in each of top_terms, kept_terms_preview and removed_terms_preview.

        Returns:
            Dict[str, Any]: result_id, threshold, n_terms, n_kept, n_removed, top_terms (term and
            variance, highest first), kept_terms_preview and removed_terms_preview. The full ranking
            is stored for visualize_result_tool.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix yet. Please call build_dtm_tool first."}
        if threshold < 0 or top_n < 1:
            return {"error": "threshold must be >= 0 and top_n >= 1."}

        report = variance_filter(session.feature_matrix, session.feature_names, threshold=threshold)
        kept = report.loc[report["keep"], "feature"].tolist()
        removed = report.loc[~report["keep"], "feature"].tolist()
        # Term first, number to plot second, already sorted: what visualize_result_tool expects.
        chart_report = report[["feature", "variance"]]

        result_id = session.next_result_id("variance")
        session.artifacts[f"{result_id}_kept_terms"] = kept
        session.artifacts[f"{result_id}_removed_terms"] = removed
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "n_terms": int(len(report)),
            "n_kept": len(kept),
            "n_removed": len(removed),
            "top_terms": [
                {"term": str(r.feature), "variance": round(float(r.variance), 6)}
                for r in report.head(top_n).itertuples()
            ],
            "kept_terms_preview": kept[:top_n],
            "removed_terms_preview": removed[:top_n],
        }
        session.store_result("variance", {"threshold": threshold, "top_n": top_n}, summary, full_report=chart_report)
        return summary

    @tool
    def pearson_filter_tool(target_class: str, top_n: int = 20) -> Dict[str, Any]:
        """Pearson correlation between each term's count and one target category (one-vs-rest:
        1 for documents of target_class, 0 for the others). Linear relationship; ranks the
        terms by absolute correlation. Needs build_dtm_tool first.

        Args:
            target_class: category_name to correlate against, e.g. "sci.med".
            top_n: how many of the strongest terms to list in the summary.

        Returns:
            Dict[str, Any]: result_id, target_class, n_terms and top_terms (term, pearson_r and
            p_value, strongest absolute correlation first). The full ranking is stored for
            visualize_result_tool.
        """
        y, error = _binary_target(target_class)
        if error:
            return error
        if top_n < 1:
            return {"error": "top_n must be at least 1."}

        report = pearson_filter(session.feature_matrix, y, session.feature_names)
        chart_report = report[["feature", "pearson_r"]]

        result_id = session.next_result_id("pearson")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "n_terms": int(len(report)),
            "top_terms": [
                {"term": str(r.feature), "pearson_r": round(float(r.pearson_r), 4), "p_value": float(f"{r.pearson_p:.3g}")}
                for r in report.head(top_n).itertuples()
            ],
        }
        session.store_result("pearson", {"target_class": target_class, "top_n": top_n}, summary, full_report=chart_report)
        return summary

    @tool
    def spearman_filter_tool(target_class: str, top_n: int = 20, max_terms: int = 1500) -> Dict[str, Any]:
        """Spearman (rank) correlation between each term's count and one target category
        (one-vs-rest). Monotonic rather than strictly linear relationship, ranked by absolute
        correlation. On a large vocabulary only the max_terms highest-variance terms are
        evaluated, to stay fast. Needs build_dtm_tool first.

        Args:
            target_class: category_name to correlate against, e.g. "sci.med".
            top_n: how many of the strongest terms to list in the summary.
            max_terms: only the max_terms highest-variance terms are evaluated when the vocabulary is larger.

        Returns:
            Dict[str, Any]: result_id, target_class, n_terms_evaluated and top_terms (term, spearman_r
            and p_value, strongest absolute correlation first). The full ranking is stored for
            visualize_result_tool.
        """
        y, error = _binary_target(target_class)
        if error:
            return error
        if top_n < 1 or max_terms < 1:
            return {"error": "top_n and max_terms must be at least 1."}

        report = spearman_filter(session.feature_matrix, y, session.feature_names, max_terms=max_terms)
        chart_report = report[["feature", "spearman_r"]]

        result_id = session.next_result_id("spearman")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "n_terms_evaluated": int(len(report)),
            "top_terms": [
                {"term": str(r.feature), "spearman_r": round(float(r.spearman_r), 4), "p_value": float(f"{r.spearman_p:.3g}")}
                for r in report.head(top_n).itertuples()
            ],
        }
        session.store_result("spearman", {"target_class": target_class, "top_n": top_n, "max_terms": max_terms}, summary, full_report=chart_report)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
