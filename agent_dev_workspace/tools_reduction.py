from typing import Any, Dict, List

import numpy as np
from langchain_core.tools import tool
from matplotlib.figure import Figure

from agent_pipeline.session_state import SessionState

_METHODS = ("pca", "tsne", "umap")
_MAX_COORDINATES_IN_SUMMARY = 50


def make_tools(session: SessionState) -> List[Any]:
    """Factory function to build the dimensionality-reduction and label-encoding tools bound to the session."""

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: int = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> Dict[str, Any]:
        """Reduce the global document-term matrix to 2 or 3 dimensions with PCA, t-SNE or UMAP
        and ALWAYS draw a scatter plot of the result, colored by category. Needs build_dtm_tool
        first. t-SNE and UMAP can take a few minutes on a large dataset.

        Args:
            method: "pca", "tsne" or "umap".
            n_components: number of output dimensions, 2 or 3 (the scatter plot uses the first two).
            random_state: seed so that t-SNE and UMAP give the same result on every run (PCA ignores it).
            perplexity: t-SNE only; must be smaller than the number of documents.
            n_neighbors: UMAP only; must be smaller than the number of documents.

        Returns:
            Dict[str, Any]: result_id, method, n_documents, n_components, coordinates (every document,
            only when there are at most 50 documents, otherwise coordinates_preview with the first 5),
            and for PCA explained_variance_ratio. The full coordinates are stored in session.artifacts.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix yet. Please call build_dtm_tool first."}
        if method not in _METHODS:
            return {"error": f"method must be one of {list(_METHODS)}."}
        if n_components not in (2, 3):
            return {"error": "n_components must be 2 or 3."}

        matrix = session.feature_matrix
        n_documents = matrix.shape[0]
        if method == "tsne" and perplexity >= n_documents:
            return {"error": f"perplexity ({perplexity}) must be smaller than the number of documents ({n_documents})."}
        if method == "umap" and n_neighbors >= n_documents:
            return {"error": f"n_neighbors ({n_neighbors}) must be smaller than the number of documents ({n_documents})."}

        # PCA, t-SNE and UMAP need a dense array; halve its memory on big matrices.
        dense = matrix.toarray()
        if dense.size > 20_000_000:
            dense = dense.astype(np.float32)

        extra: Dict[str, Any] = {}
        if method == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=n_components)
            coordinates = reducer.fit_transform(dense)
            extra["explained_variance_ratio"] = [round(float(v), 4) for v in reducer.explained_variance_ratio_]
        elif method == "tsne":
            from sklearn.manifold import TSNE

            coordinates = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(dense)
        else:
            import umap

            coordinates = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state).fit_transform(dense)

        labels = np.asarray(session.labels)
        fig = Figure(figsize=(9, 7))
        ax = fig.subplots()
        for category in session.categories:
            mask = labels == category
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], label=str(category), alpha=0.6, s=18)
        ax.set_title(f"{method.upper()} of the document-term matrix, colored by category")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(loc="best")
        fig.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("reduce")
        session.artifacts[f"{method}_coordinates"] = coordinates
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "method": method,
            "n_documents": int(n_documents),
            "n_components": n_components,
            **extra,
        }
        rounded = np.round(coordinates, 4).tolist()
        if n_documents <= _MAX_COORDINATES_IN_SUMMARY:
            summary["coordinates"] = rounded
        else:
            summary["coordinates_preview"] = rounded[:5]
        session.store_result(
            "reduce",
            {"method": method, "n_components": n_components, "random_state": random_state, "perplexity": perplexity, "n_neighbors": n_neighbors},
            summary,
        )
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encode the category labels of the working DataFrame: one 0/1 column per
        category (sorted alphabetically) and one row per document, with a single 1 per row.
        The matrix is stored in session.artifacts["binarized_labels"].

        Returns:
            Dict[str, Any]: result_id, classes (the column order), shape, counts_per_class and a
            preview of the first 5 rows.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please call load_dataset_tool first."}

        labels = session.dataframe["category_name"].astype(str).to_numpy()
        classes = sorted(set(labels.tolist()))
        encoded = (labels[:, None] == np.asarray(classes)[None, :]).astype(int)

        session.artifacts["binarized_labels"] = encoded
        session.artifacts["binarized_classes"] = classes

        result_id = session.next_result_id("binarize")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "shape": [int(encoded.shape[0]), int(encoded.shape[1])],
            "counts_per_class": {c: int(n) for c, n in zip(classes, encoded.sum(axis=0))},
            "preview": [
                {"category_name": str(lab), "encoded": row.tolist()} for lab, row in zip(labels[:5], encoded[:5])
            ],
        }
        session.store_result("binarize", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
