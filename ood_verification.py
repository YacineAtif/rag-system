import numpy as np
import torch

from types import SimpleNamespace


from types import SimpleNamespace


from typing import List
from sentence_transformers import util


class OODVerificationAgent:
    def __init__(self, config, neo4j_driver, text_embedder, document_store, text_processor):

        default = SimpleNamespace(enabled=False, similarity_threshold=0.65, min_neo4j_relations=1)
        self.config = config
        self.ood_config = getattr(config, "ood", default)


        default = SimpleNamespace(enabled=False, similarity_threshold=0.65, min_neo4j_relations=1)
        self.config = config
        self.ood_config = getattr(config, "ood", default)

        self.config = config


        self.neo4j_driver = neo4j_driver
        self.text_embedder = text_embedder
        self.document_store = document_store
        self.text_processor = text_processor
        self.domain_centroid = self._compute_domain_centroid()

    def _compute_domain_centroid(self):
        """Compute average embedding of all documents as domain centroid"""

        try:
            all_docs = self.document_store.filter_documents()
        except Exception:
            return None
        if not all_docs:
            return None
        embeddings = [np.array(doc.embedding, dtype=np.float32)
                      for doc in all_docs
                      if getattr(doc, "embedding", None) is not None]
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0).astype(np.float32)


        all_docs = self.document_store.filter_documents()
        if not all_docs:
            return None
        embeddings = [doc.embedding for doc in all_docs if getattr(doc, "embedding", None) is not None]
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)



    def embedding_similarity_check(self, query: str) -> bool:
        """Check if query is in domain embedding space"""
        if self.domain_centroid is None:
            return True


        try:
            result = self.text_embedder.run(text=query)
            query_embedding = np.array(result["embedding"], dtype=np.float32)
            similarity = util.pytorch_cos_sim(
                torch.tensor(query_embedding, dtype=torch.float32),
                torch.tensor(self.domain_centroid, dtype=torch.float32)
            ).item()
        except Exception as e:
            print(f"⚠️ Embedding check failed: {e}")
            return True
        threshold = getattr(self.ood_config, "similarity_threshold", 0.65)


        result = self.text_embedder.run(text=query)
        query_embedding = result["embedding"]
        similarity = util.pytorch_cos_sim(
            torch.tensor(query_embedding),
            torch.tensor(self.domain_centroid)
        ).item()
        threshold = self.config["ood"].get("similarity_threshold", 0.65)


        return similarity >= threshold

    def neo4j_knowledge_check(self, query: str) -> bool:
        """Verify entities exist in knowledge graph"""

        try:
            entities = self.text_processor.extract_entities(query)
        except Exception:
            entities = []
        if not entities:
            return False
        min_relations = getattr(self.ood_config, "min_neo4j_relations", 1)
        try:
            with self.neo4j_driver.session() as session:
                node_result = session.run("MATCH (n) RETURN count(n) AS node_count")
                node_count = node_result.single()["node_count"]
                if node_count == 0:
                    return True
                for entity in entities:
                    result = session.run(
                        "MATCH (e:Entity {name: $name})-[r]-() "
                        "RETURN count(r) >= $min_relations AS has_relations",
                        name=entity,
                        min_relations=min_relations
                    )
                    if result.single()["has_relations"]:
                        return True
        except Exception as e:
            print(f"⚠️ Neo4j check failed: {e}")
            return True


        entities = self.text_processor.extract_entities(query)
        if not entities:
            return False
        min_relations = self.config["ood"].get("min_neo4j_relations", 1)
        with self.neo4j_driver.session() as session:
            node_result = session.run("MATCH (n) RETURN count(n) AS node_count")
            node_count = node_result.single()["node_count"]
            if node_count == 0:
                return True
            for entity in entities:
                result = session.run(
                    "MATCH (e:Entity {name: $name})-[r]-() "
                    "RETURN count(r) >= $min_relations AS has_relations",
                    name=entity,
                    min_relations=min_relations
                )
                if result.single()["has_relations"]:
                    return True


        return False

    def verify(self, query: str) -> bool:
        """Execute full verification pipeline"""

        if not getattr(self.ood_config, "enabled", False):
            return True
        try:
            if not self.embedding_similarity_check(query):
                return False
            return self.neo4j_knowledge_check(query)
        except Exception as e:
            print(f"⚠️ OOD verification error: {e}")
            return True


        if not self.config.get("ood", {}).get("enabled", True):
            return True
        if not self.embedding_similarity_check(query):
            return False
        return self.neo4j_knowledge_check(query)



    def auto_adjust_threshold(self, sample_queries: List[str]):
        """Dynamically set similarity threshold based on sample queries"""
        if self.domain_centroid is None:
            return
        similarities = []
        for query in sample_queries:

            try:
                result = self.text_embedder.run(text=query)
                query_embedding = np.array(result["embedding"], dtype=np.float32)
                similarity = util.pytorch_cos_sim(
                    torch.tensor(query_embedding, dtype=torch.float32),
                    torch.tensor(self.domain_centroid, dtype=torch.float32)
                ).item()
                similarities.append(similarity)
            except Exception as e:
                print(f"⚠️ Threshold adjustment failed for query '{query}': {e}")


            result = self.text_embedder.run(text=query)
            query_embedding = result["embedding"]
            similarity = util.pytorch_cos_sim(
                torch.tensor(query_embedding),
                torch.tensor(self.domain_centroid)
            ).item()
            similarities.append(similarity)


        if not similarities:
            return
        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        new_threshold = max(0.5, mean_sim - std_sim)

        setattr(self.ood_config, "similarity_threshold", new_threshold)
        print(f"🔧 Auto-adjusted similarity threshold to {new_threshold:.2f}")


        setattr(self.ood_config, "similarity_threshold", new_threshold)
        print(f"🔧 Auto-adjusted similarity threshold to {new_threshold:.2f}")

        self.config.ood.similarity_threshold = new_threshold
        print(
            f"\ud83d\udd27 Auto-adjusted similarity threshold to {new_threshold:.2f}"
        )


