import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from agent_pipeline.session_state import SessionState

_FILTERING_METHODS = ("variance", "tfidf", "term_frequency")
_ALGORITHMS = ("fpgrowth", "topk", "maxfpgrowth")


def _filter_terms(term_doc: pd.DataFrame, filtering_method: str) -> List[str]:
    """Same vocabulary filters as the Master notebook (helpers/text_filtering_helpers.py):
    variance and term_frequency keep the middle 90% (5th to 95th percentile) of the metric,
    tfidf keeps everything at or above the 20th percentile of the mean TF-IDF score."""
    if filtering_method == "tfidf":
        scores = np.asarray(TfidfTransformer().fit_transform(term_doc.values).mean(axis=0)).flatten()
        metric = pd.Series(scores, index=term_doc.columns)
        return metric[metric >= np.percentile(metric, 20)].index.tolist()

    metric = term_doc.var(axis=0) if filtering_method == "variance" else term_doc.sum(axis=0)
    lower, upper = np.percentile(metric, 5), np.percentile(metric, 95)
    return metric[(metric >= lower) & (metric <= upper)].index.tolist()


def make_tools(session: SessionState) -> List[Any]:
    """Factory function to build the frequent-pattern mining tool bound to the session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: int = None,
        k: int = None,
    ) -> Dict[str, Any]:
        """Mine frequent word patterns (groups of words that appear together in many documents)
        in ONE category, following the Master notebook: build a fresh document-term matrix of that
        category only (CountVectorizer with English stop words removed), keep the terms that pass
        the chosen filtering_method, turn the documents into a transactional database and mine it
        with PAMI. Call it once per category. It is independent of build_dtm_tool. Having no term
        survive the filter is a normal outcome and returns an empty list of patterns.

        Args:
            category_name: the category to mine, e.g. "sci.med"; other categories are excluded.
            filtering_method: "variance" or "term_frequency" (keep the middle 90% of terms by
                variance or by total count) or "tfidf" (keep the terms at or above the 20th
                percentile of mean TF-IDF).
            algorithm: "fpgrowth" (all patterns with support >= min_sup), "maxfpgrowth" (only the
                maximal ones, support >= min_sup) or "topk" (the k most frequent patterns).
            min_sup: minimum number of documents a pattern must appear in. Required for "fpgrowth"
                and "maxfpgrowth".
            k: how many patterns to keep. Required for "topk".

        Returns:
            Dict[str, Any]: result_id, category_name, filtering_method, algorithm, n_documents,
            n_terms_before_filter, n_terms_after_filter, kept_terms_preview, n_patterns and patterns
            (up to 20, highest support first, each with its terms and support), or a note when no
            term survived the filter.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please call load_dataset_tool first."}
        if filtering_method not in _FILTERING_METHODS:
            return {"error": f"filtering_method must be one of {list(_FILTERING_METHODS)}."}
        if algorithm not in _ALGORITHMS:
            return {"error": f"algorithm must be one of {list(_ALGORITHMS)}."}
        if algorithm == "topk" and (k is None or k < 1):
            return {"error": "algorithm 'topk' needs k (a positive integer)."}
        if algorithm != "topk" and (min_sup is None or min_sup < 1):
            return {"error": f"algorithm '{algorithm}' needs min_sup (a positive integer)."}

        df = session.dataframe
        category_df = df[df["category_name"].astype(str) == category_name]
        if category_df.empty:
            return {
                "error": f"Unknown category_name '{category_name}'.",
                "available_categories": sorted(df["category_name"].astype(str).unique().tolist()),
            }

        # Fresh, category-local matrix: a different vocabulary than build_dtm_tool's global one.
        vectorizer = CountVectorizer(stop_words="english")
        counts = vectorizer.fit_transform(category_df["text"].fillna("").astype(str))
        term_doc = pd.DataFrame(counts.toarray(), columns=vectorizer.get_feature_names_out())

        kept_terms = _filter_terms(term_doc, filtering_method) if term_doc.shape[1] else []

        result_id = session.next_result_id("patterns")
        args = {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_documents": int(len(category_df)),
            "n_terms_before_filter": int(term_doc.shape[1]),
            "n_terms_after_filter": len(kept_terms),
            "kept_terms_preview": kept_terms[:20],
        }
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        if not kept_terms:
            summary.update({"n_patterns": 0, "patterns": [], "note": "no terms survived filtering"})
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns", args, summary)
            return summary

        patterns_df = _mine(term_doc[kept_terms], algorithm, min_sup, k)
        patterns = [
            {"terms": [t for t in str(row.Patterns).split() if t], "support": int(row.Support)}
            for row in patterns_df.sort_values("Support", ascending=False, kind="stable").itertuples()
        ]
        summary.update({"n_patterns": len(patterns), "patterns": patterns[:20]})
        if not patterns:
            summary["note"] = "the mining algorithm returned no patterns for these settings"

        session.artifacts[artifact_key] = patterns
        report = pd.DataFrame(
            {"pattern": [" ".join(p["terms"]) for p in patterns], "support": [p["support"] for p in patterns]}
        )
        session.store_result("mine_patterns", args, summary, full_report=report)
        return summary

    return [mine_patterns_tool]


def _mine(filtered: pd.DataFrame, algorithm: str, min_sup: int, k: int) -> pd.DataFrame:
    """Write the filtered matrix as a PAMI transactional database (one transaction per document,
    items are the terms with count >= 1) to a temporary file and mine it."""
    from PAMI.extras.convert.DF2DB import DF2DB

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "transactions.csv")
        DF2DB(filtered).convert2TransactionalDatabase(db_path, ">=", 1)

        if algorithm == "fpgrowth":
            from PAMI.frequentPattern.basic import FPGrowth

            miner = FPGrowth.FPGrowth(iFile=db_path, minSup=min_sup, sep="\t")
        elif algorithm == "maxfpgrowth":
            from PAMI.frequentPattern.maximal import MaxFPGrowth

            miner = MaxFPGrowth.MaxFPGrowth(iFile=db_path, minSup=min_sup, sep="\t")
        else:
            from PAMI.frequentPattern.topk import FAE

            miner = FAE.FAE(iFile=db_path, k=k, sep="\t")
        miner.mine()
        return miner.getPatternsAsDataFrame()
