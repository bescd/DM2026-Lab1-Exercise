import os
from typing import Any, Dict, List, Optional

import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from matplotlib.figure import Figure

from agent_pipeline.session_state import SessionState


def _resolve_within_repo(path: str) -> Optional[str]:
    """Resolve `path` relative to the repo root (the notebook's cwd),
    refusing anything that escapes it."""
    root = os.path.abspath(os.getcwd())
    target = os.path.abspath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        return None
    return target


def _invalidate_derived(session: SessionState) -> bool:
    """Drop the document-term matrix if one exists: after the rows of
    session.dataframe change, its rows no longer line up with the
    dataframe's. Returns True if something was cleared."""
    had_matrix = session.feature_matrix is not None
    session.feature_matrix = None
    session.feature_names = None
    session.labels = None
    session.categories = []
    session.artifacts.pop("count_vectorizer", None)
    return had_matrix


def _word_tokenize(text: str) -> List[str]:
    """nltk.word_tokenize on lowercased text; falls back to NLTK's own
    word tokenizer (the same Treebank-style rules, minus the sentence
    splitter that needs the 'punkt' download) if that data is missing."""
    try:
        return nltk.word_tokenize(text.lower())
    except LookupError:
        from nltk.tokenize import NLTKWordTokenizer

        return NLTKWordTokenizer().tokenize(text.lower())


def make_tools(session: SessionState) -> List[Any]:
    """Factory function to build and return data tools bound to the session."""

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Check for missing or empty text values in the loaded dataset.

        Examines session.dataframe['text'] for nulls or whitespace-only strings.

        Returns:
            Dict[str, Any]: A dictionary containing result_id, n_missing, missing_indices, and total_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please call load_dataset_tool first."}

        df = session.dataframe
        text_col = df["text"]

        # Identify null or empty/whitespace-only strings
        missing_mask = text_col.isna() | (text_col.astype(str).str.strip() == "")
        missing_indices = df[missing_mask].index.tolist()
        n_missing = len(missing_indices)
        total_rows = len(df)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "n_missing": n_missing,
            "missing_indices": missing_indices,
            "total_rows": total_rows,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize each document's text into lowercase unigrams with
        nltk.word_tokenize and store them in session.dataframe["unigrams"].

        Run it after load_dataset_tool. It does not change the text itself.

        Returns:
            Dict[str, Any]: result_id, n_documents, total_tokens, avg_tokens_per_document,
            n_empty_documents and a preview of the first tokens of the first 3 documents.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please call load_dataset_tool first."}

        df = session.dataframe
        unigrams = df["text"].fillna("").astype(str).apply(_word_tokenize)
        df["unigrams"] = unigrams
        session.artifacts["tokens"] = unigrams.tolist()

        lengths = unigrams.apply(len)
        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": int(len(df)),
            "total_tokens": int(lengths.sum()),
            "avg_tokens_per_document": round(float(lengths.mean()), 2) if len(df) else 0.0,
            "n_empty_documents": int((lengths == 0).sum()),
            "preview": [tokens[:10] for tokens in unigrams.head(3)],
        }
        session.store_result("tokenize", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """List the files and folders inside a directory of the repo, so a
        dataset path can be found before loading it. Hidden entries and
        __pycache__ are left out; paths outside the repo are refused.

        Args:
            subdirectory: folder to list, relative to the repo root (default "." is the root itself).

        Returns:
            Dict[str, Any]: result_id, subdirectory, directories (sorted) and files (sorted).
        """
        target = _resolve_within_repo(subdirectory)
        if target is None:
            return {"error": f"'{subdirectory}' is outside the repo."}
        if not os.path.isdir(target):
            return {"error": f"Directory not found within the repo: '{subdirectory}'."}

        directories, files = [], []
        for name in sorted(os.listdir(target)):
            if name.startswith(".") or name == "__pycache__":
                continue
            (directories if os.path.isdir(os.path.join(target, name)) else files).append(name)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }
        session.store_result("list_files", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5, start: int = 0, max_chars: int = 200) -> Dict[str, Any]:
        """Show a few rows of the working DataFrame (text, category and category_name),
        with long texts cut short, so the loaded data can be eyeballed.

        Args:
            n_rows: how many rows to show.
            start: position of the first row to show (0 is the first row).
            max_chars: texts longer than this many characters are cut and end with "...".

        Returns:
            Dict[str, Any]: result_id, shape, columns, start, and rows (a list with one dict per row).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please call load_dataset_tool first."}
        if n_rows < 1 or start < 0 or max_chars < 1:
            return {"error": "n_rows and max_chars must be at least 1, and start at least 0."}

        df = session.dataframe
        rows = []
        for position in range(start, min(start + n_rows, len(df))):
            row = df.iloc[position]
            text = str(row["text"])
            rows.append(
                {
                    "position": position,
                    "index": df.index[position].item() if hasattr(df.index[position], "item") else df.index[position],
                    "category_name": str(row["category_name"]),
                    "text": text if len(text) <= max_chars else text[:max_chars] + "...",
                    "text_length": len(text),
                }
            )

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": [str(c) for c in df.columns],
            "start": start,
            "rows": rows,
        }
        session.store_result("inspect", {"n_rows": n_rows, "start": start, "max_chars": max_chars}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check the working DataFrame for duplicate documents (same text and same
        category) and optionally drop them. Dropping uses keep=False, like the Master
        notebook: every copy of a duplicated document is removed, including the first one.
        Dropping invalidates the document-term matrix, so rebuild it afterwards.

        Args:
            drop: if True, remove every duplicated document; if False, only report them.

        Returns:
            Dict[str, Any]: result_id, n_duplicates (extra copies, keep="first" count),
            n_rows_involved (all rows that belong to a duplicated group), n_rows_before,
            n_rows_after and dropped.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please call load_dataset_tool first."}

        df = session.dataframe
        # Only the plain columns: "unigrams" holds lists, which can't be hashed.
        subset = [c for c in ("text", "category", "category_name") if c in df.columns]
        n_duplicates = int(df.duplicated(subset=subset).sum())
        n_rows_involved = int(df.duplicated(subset=subset, keep=False).sum())
        n_before = int(len(df))

        matrix_cleared = False
        if drop and n_rows_involved:
            session.dataframe = df.drop_duplicates(subset=subset, keep=False)
            matrix_cleared = _invalidate_derived(session)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "n_duplicates": n_duplicates,
            "n_rows_involved": n_rows_involved,
            "n_rows_before": n_before,
            "n_rows_after": int(len(session.dataframe)),
            "dropped": bool(drop and n_rows_involved),
        }
        if matrix_cleared:
            summary["note"] = "The document-term matrix was cleared because rows were dropped; run build_dtm_tool again."
        session.store_result("check_duplicates", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: int = 1) -> Dict[str, Any]:
        """Replace the working DataFrame by a random sample of n documents
        (pandas DataFrame.sample), to get a smaller working set. The original rows are
        kept in session.artifacts["dataframe_before_sampling"]. Sampling invalidates the
        document-term matrix, so build it after sampling.

        Args:
            n: number of documents to keep; must not exceed the current number of documents.
            random_state: seed of the random sample, for reproducible results.

        Returns:
            Dict[str, Any]: result_id, n_before, n_after, random_state, sampled_indices
            (original row labels of the first 20 sampled rows) and counts_per_category.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please call load_dataset_tool first."}
        df = session.dataframe
        if n < 1 or n > len(df):
            return {"error": f"n must be between 1 and {len(df)}, got {n}."}

        session.artifacts["dataframe_before_sampling"] = df
        sample = df.sample(n=n, random_state=random_state)
        session.dataframe = sample
        _invalidate_derived(session)

        result_id = session.next_result_id("sample")
        summary = {
            "result_id": result_id,
            "n_before": int(len(df)),
            "n_after": int(len(sample)),
            "random_state": random_state,
            "sampled_indices": [int(i) for i in sample.index[:20]],
            "counts_per_category": {str(k): int(v) for k, v in sample["category_name"].value_counts().items()},
        }
        session.store_result("sample", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Describe the document length (number of characters) with pandas .describe(),
        overall and per category, and draw a box plot of the length by category.

        Returns:
            Dict[str, Any]: result_id, overall (count, mean, std, min, 25%, 50%, 75%, max)
            and per_category with the same statistics for each category_name.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please call load_dataset_tool first."}

        df = session.dataframe
        text_length = df["text"].fillna("").astype(str).apply(len)

        def _stats(series: pd.Series) -> Dict[str, float]:
            return {str(k): round(float(v), 4) for k, v in series.describe().items()}

        per_category = {
            str(name): _stats(group) for name, group in text_length.groupby(df["category_name"].astype(str))
        }

        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()
        plot_df = pd.DataFrame({"category_name": df["category_name"].astype(str).values, "text_length": text_length.values})
        sns.boxplot(data=plot_df, x="category_name", y="text_length", ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Text Length Distribution by Category")
        ax.set_ylabel("Text Length (characters)")
        ax.set_xlabel("Category")
        fig.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("describe")
        summary = {"result_id": result_id, "overall": _stats(text_length), "per_category": per_category}
        session.store_result("describe", {}, summary)
        return summary

    return [
        check_missing_tool,
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
