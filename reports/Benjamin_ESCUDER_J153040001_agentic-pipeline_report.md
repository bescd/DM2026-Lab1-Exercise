# Agentic Pipeline Report

I ran my own tools on the Reddit stock sentiment dataset.

I loaded the data with **load_dataset_tool**. The file is newdataset/Reddit-stock-sentiment.csv, text column "text", label column "label". It has 847 posts. 423 are neutral (0.0), 315 are negative (-1.0), 109 are positive (1.0).

I checked for duplicates with **check_duplicates_tool**. It found 23 duplicate texts, 34 rows total. I dropped them with keep=False. The dataset went from 847 to 813 rows. I also ran check_missing_tool: no document had empty text.

I ran **describe_data_tool** (result describe_8) on the 813 rows. Average text length is about 148 characters. The box plot is at ../plots/Benjamin_ESCUDER_agentic-pipeline_plot_describe_8.png. All three classes have most posts under 200 characters, with a few very long outliers. The longest post is about 2750 characters, in the neutral class. The negative class has an outlier near 2040 characters, the positive class near 2460.

813 rows was still a lot, so I used **sample_data_tool** to get a smaller working set. I picked n=200, random_state=1. The sample has 98 neutral, 78 negative, 24 positive posts. The positive class is small in this sample.

I ran **tokenize_tool** on the sample. It found 7231 tokens in total, about 36 tokens per document on average. No document was empty.

I ran **binarize_labels_tool** to one-hot encode the three classes. The class order is -1.0, 0.0, 1.0. The counts match the sample: 78, 98, 24.

I built the document-term matrix with **build_dtm_tool**, no stop words removed. I checked the most frequent words with **term_frequency_tool** (result term_frequency_11, see ../plots/Benjamin_ESCUDER_agentic-pipeline_plot_turn80.png). The top words are common English words: "the" (about 235 times), "to" (about 165), "and" (about 135), "is" (about 100), "of" (about 90). None of these are useful for sentiment on their own, since I did not remove stop words at this step.

I looked at a slice of the matrix with **dtm_heatmap_tool** (../plots/Benjamin_ESCUDER_agentic-pipeline_plot_dtm_heatmap_12.png). The first 15 terms in alphabetical order are numbers, like "00", "08", "10000". None of them appear in the first 15 sampled documents, so the heatmap was all zero. I ran it again with 40 terms (../plots/Benjamin_ESCUDER_agentic-pipeline_plot_dtm_heatmap_13.png). Still all zero: the next 25 terms are more numbers and odd tokens like "22rate", "25bps", "2c_books". This happens because the tool reads terms in vocabulary order, not by frequency, and numbers sort before letters.

I ran **feature_correlation_matrix_tool** (../plots/Benjamin_ESCUDER_agentic-pipeline_plot_feature_correlation_19.png). The top 20 terms by variance are mostly stop words, matching term_frequency_11. The strongest correlation is between "you" and "your" (0.92), then "you" and "not" (0.84). This makes sense, these words often appear together in the same sentence. One term stands out: "graphite". It has high variance but low correlation with every other term (around 0.0 to 0.35), so it is probably tied to one specific post, not a general pattern.

I ran **cosine_similarity_tool** between document 0 and document 2. Both are in the neutral category, but their similarity is 0.0. They share no common words. This shows that same category does not mean same words, especially with short Reddit comments.

Before going further, I rebuilt the document-term matrix with **build_dtm_tool**, this time with stop_words="english". The new matrix has 200 documents and 1759 terms, with 2729 non-zero entries out of 351800 total, so it is 99.22% sparse. This is the matrix used for everything below.

I ran **variance_filter_tool** with threshold=0.01. Out of 1759 terms, 331 passed the threshold and 1428 were removed. The highest-variance term is "graphite" (0.40), then "like" (0.25), "company" (0.24), "money" (0.21), "trump" (0.19). "Graphite" being on top again confirms what I saw in feature_correlation_19: it is a strong but isolated signal, probably from one post talking a lot about graphite.

I ran **pearson_filter_tool** with target_class="1.0" (positive posts) and top_n=15. The strongest positive terms are "tldr", "green", "follow", "downtrend", "potential", "trend" (all r=0.27), then "mean", "handle", "stuff", "info", "means" (r=0.21). These r values are not very strong, and the positive class only has 24 posts in this sample, so I would not trust this correlation much. A word appearing in 2 or 3 of the 24 positive posts can already look "correlated" by chance with so few examples.

I ran **spearman_filter_tool** with target_class="-1.0" (negative posts) and top_n=15, out of 1500 evaluated terms. Top terms are "xi", "americans", "sure", "treasuries", "safe", "great" (all r=0.18), then "japan", "makes", "free", "shit", "allies", "dead", "trying", "fucking", "end" (r=0.15). These are different terms from the Pearson result for class 1.0, which makes sense since this is a different class. Some terms here ("xi", "americans", "japan", "treasuries", "allies") point to posts about geopolitics and trade, which the negative class seems to talk about more.

I ran **mine_patterns_tool** for category "1.0" with filtering_method="tfidf", algorithm="fpgrowth", min_sup=3. The category has 24 documents. After tf-idf filtering, 259 of 299 terms were kept. Only 3 patterns reached the minimum support: ["just"] (support 4), ["new"] (support 3), ["like"] (support 3). With only 24 documents, there is not much room for frequent multi-word patterns to show up.

Finally, I ran **reduce_dimensions_tool** with method="pca" (result reduce_22, see ../plots/Benjamin_ESCUDER_agentic-pipeline_plot_reduce_22.png). The two components explain very little of the variance: 8.56% and 6.11%, so together just under 15%. In the plot, the three classes overlap heavily, with no clear separation. This matches what I already saw with cosine_similarity_tool: two posts in the same category can share no words at all, so a simple word-count-based 2D projection cannot cleanly separate sentiment classes in this dataset.
