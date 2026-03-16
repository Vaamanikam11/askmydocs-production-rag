# CI Test Doc

AskMyDocs is a production-grade RAG application.

Hybrid retrieval combines BM25 keyword search with vector semantic search to improve recall across diverse query types.

Cross-encoder reranking improves precision by scoring each (query, chunk) pair jointly.

If evidence is weak, the system should abstain rather than hallucinate.

A golden dataset is a set of verified Q/A pairs used for regression testing.

CI gating blocks merges when retrieval quality drops below a threshold.
