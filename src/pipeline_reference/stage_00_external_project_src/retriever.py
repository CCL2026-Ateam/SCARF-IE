"""Few-shot exemplar retrieval from the pool.

The default retriever uses lightweight lexical similarity (token Jaccard).  An
optional embedding retriever implements SRAG-style semantic retrieval with
SentenceTransformer when the dependency/model is available.
"""
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


def tokenize(text: str) -> set:
    return set(t.lower() for t in _TOKEN_RE.findall(text))


class ExemplarRetriever:
    def __init__(self, pool: List[Dict[str, Any]], prefer_relations: bool = True):
        self.pool = pool
        self.pool_tokens = [tokenize(r["text"]) for r in pool]
        self.prefer_relations = prefer_relations

    def retrieve(self, text: str, k: int = 4, ensure_rel: int = 2) -> List[Dict[str, Any]]:
        q = tokenize(text)
        if not q:
            return self.pool[:k]
        scored = []
        for i, pt in enumerate(self.pool_tokens):
            if not pt:
                continue
            inter = len(q & pt)
            if inter == 0:
                continue
            union = len(q | pt)
            jac = inter / union
            scored.append((jac, i))
        scored.sort(reverse=True)

        chosen = []
        used = set()
        # First, ensure a few exemplars that actually contain relations (RE is 60% of score)
        if ensure_rel > 0:
            cnt = 0
            for jac, i in scored:
                if cnt >= ensure_rel:
                    break
                if self.pool[i]["relations"]:
                    chosen.append(i)
                    used.add(i)
                    cnt += 1
        # Fill the rest by similarity
        for jac, i in scored:
            if len(chosen) >= k:
                break
            if i in used:
                continue
            chosen.append(i)
            used.add(i)
        # pad if needed
        if len(chosen) < k:
            for i in range(len(self.pool)):
                if len(chosen) >= k:
                    break
                if i not in used:
                    chosen.append(i)
                    used.add(i)
        # order by similarity ascending so the most similar is last (closest to query)
        sim_map = {i: j for j, i in scored}
        chosen.sort(key=lambda i: sim_map.get(i, 0.0))
        return [self.pool[i] for i in chosen[:k]]


class SemanticExemplarRetriever:
    """SRAG retriever backed by sentence-transformers embeddings.

    The class is intentionally optional: imports happen lazily so environments
    without torch/sentence-transformers can keep using the lexical retriever.
    Embeddings are cached as plain JSON to avoid requiring numpy at load time.
    """

    def __init__(
        self,
        pool: List[Dict[str, Any]],
        model_path: str,
        cache_path: str = None,
        prefer_relations: bool = True,
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "semantic SRAG requires sentence-transformers/torch. "
                "Install them or use --retriever lexical."
            ) from exc

        self.pool = pool
        self.prefer_relations = prefer_relations
        self.model_path = model_path
        self.cache_path = Path(cache_path) if cache_path else None
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else None
        except Exception:
            device = None
        self.model = SentenceTransformer(model_path, device=device)
        self.embeddings = self._load_or_build_embeddings(batch_size=batch_size)

    @staticmethod
    def _as_list_vector(vec):
        if hasattr(vec, "detach"):
            vec = vec.detach()
        if hasattr(vec, "cpu"):
            vec = vec.cpu()
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return [float(x) for x in vec]

    @staticmethod
    def _normalize(vec):
        norm = math.sqrt(sum(x * x for x in vec))
        if norm <= 0:
            return vec
        return [x / norm for x in vec]

    @staticmethod
    def _dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def _cache_payload(self):
        return {
            "model_path": self.model_path,
            "texts": [item["text"] for item in self.pool],
        }

    def _load_or_build_embeddings(self, batch_size: int):
        payload = self._cache_payload()
        if self.cache_path and self.cache_path.exists():
            try:
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if cached.get("model_path") == payload["model_path"] and cached.get("texts") == payload["texts"]:
                    return cached["embeddings"]
            except Exception:
                pass

        texts = payload["texts"]
        raw = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        embeddings = [self._normalize(self._as_list_vector(vec)) for vec in raw]
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps({**payload, "embeddings": embeddings}, ensure_ascii=False),
                encoding="utf-8",
            )
        return embeddings

    def retrieve(self, text: str, k: int = 4, ensure_rel: int = 2) -> List[Dict[str, Any]]:
        q_vec = self.model.encode(text, show_progress_bar=False)
        q_vec = self._normalize(self._as_list_vector(q_vec))
        scored = [(self._dot(q_vec, emb), i) for i, emb in enumerate(self.embeddings)]
        scored.sort(reverse=True)

        chosen = []
        used = set()
        if ensure_rel > 0:
            for _score, i in scored:
                if len(chosen) >= ensure_rel:
                    break
                if self.pool[i].get("relations"):
                    chosen.append(i)
                    used.add(i)
        for _score, i in scored:
            if len(chosen) >= k:
                break
            if i not in used:
                chosen.append(i)
                used.add(i)

        score_map = {i: score for score, i in scored}
        chosen.sort(key=lambda i: score_map.get(i, 0.0))
        return [self.pool[i] for i in chosen[:k]]
