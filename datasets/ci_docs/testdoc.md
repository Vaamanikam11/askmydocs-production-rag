# CI Test Doc (AskMyDocs)

## Hybrid Retrieval
Hybrid retrieval combines BM25 keyword search with vector semantic search. BM25 helps for exact matches like error codes, IDs, and proper nouns. Vector search helps for paraphrases and semantic intent. A common approach is to retrieve top-k from BM25 and top-k from vector search, then merge using Reciprocal Rank Fusion (RRF).

## Reciprocal Rank Fusion (RRF)
RRF merges multiple ranked lists by assigning each document a fused score based on its rank in each list. This improves recall because relevant documents that appear in either list can rise in the fused ranking.

## Cross-Encoder Reranking
A cross-encoder reranker scores each (query, passage) pair jointly, enabling token-level interactions. This often improves precision compared to pure embedding similarity. The tradeoff is latency and compute, so reranking is applied only to a smaller candidate set.

## Citation Enforcement and Abstention
The assistant must answer only from retrieved evidence. If retrieved chunks do not support an answer, the system should abstain and respond that it does not have enough evidence. Every major claim should include a citation like [1] or [2], and citations must refer to provided sources.

## Golden Dataset and CI Gating
A golden dataset is a set of verified question/answer pairs used for regression testing. Offline evaluation checks retrieval quality and groundedness. CI gating blocks merges when retrieval quality drops below a threshold, preventing regressions from being merged.

## Notes (repeat for chunking stability)
Hybrid retrieval combines BM25 keyword search with vector semantic search. BM25 helps for exact matches like error codes, IDs, and proper nouns. Vector search helps for paraphrases and semantic intent. RRF merges ranked lists and improves recall. Cross-encoder reranking scores query and passage jointly, improving precision. Citation enforcement prevents hallucinations by requiring evidence-based answers. A golden dataset enables regression testing and CI gating prevents quality regressions.