from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm
from hydra.utils import get_original_cwd
import hashlib
import json
import os
class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self._openai_client: OpenAI | None = None

    def _get_openai_client(self) -> OpenAI:
        if self._openai_client is None:
            self._openai_client = OpenAI(
                api_key=self.config.llm.api.key,
                base_url=self.config.llm.api.base_url,
            )
        return self._openai_client

    def _cache_dir(self) -> str:
        try:
            original_cwd = get_original_cwd()
        except Exception:
            original_cwd = os.getcwd()
        cache_dir = self.config.executor.get("cache_dir")
        if cache_dir:
            cache_dir = str(cache_dir)
            if os.path.isabs(cache_dir):
                return cache_dir
            return os.path.join(original_cwd, cache_dir)
        return os.path.join(original_cwd, "cache")

    def _no_cache(self) -> bool:
        value = self.config.executor.get("no_cache", False)
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _load_paper_cache(self, paper_url: str, cache_dir: str) -> dict:
        paper_id = hashlib.sha256(paper_url.encode("utf-8")).hexdigest()
        path = os.path.join(cache_dir, f"{paper_id}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.debug(f"Failed to load cache file {path}: {e}")
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _save_paper_cache(self, paper_url: str, cache_dir: str, **fields) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        paper_id = hashlib.sha256(paper_url.encode("utf-8")).hexdigest()
        path = os.path.join(cache_dir, f"{paper_id}.json")

        data = self._load_paper_cache(paper_url, cache_dir)
        data.update(fields)
        data.setdefault("url", paper_url)
        data.setdefault("updated_at", datetime.now().isoformat())

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if not self.config.zotero.include_path:
            return corpus
        new_corpus = []
        logger.info(f"Selecting zotero papers matching include_path: {self.config.zotero.include_path}")
        for c in corpus:
            match_results = [glob_match(p, self.config.zotero.include_path) for p in c.paths]
            if any(match_results):
                new_corpus.append(c)
        samples = random.sample(new_corpus, min(5, len(new_corpus)))
        samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
        logger.info(f"Selected {len(new_corpus)} zotero papers:\n{samples}\n...")
        return new_corpus

    
    def run(self):
        no_cache = self._no_cache()
        cache_dir = self._cache_dir()
        today = datetime.now().date().isoformat()
        cache_html_path = os.path.join(cache_dir, f"{today}.html")

        if (not no_cache) and os.path.exists(cache_html_path):
            logger.info(f"Found cached HTML: {cache_html_path}. Skipping pipeline and re-sending.")
            with open(cache_html_path, "r", encoding="utf-8") as f:
                email_content = f.read()
            send_email(self.config, email_content)
            logger.info("Email sent successfully (from cache)")
            return

        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            paper_cache_dir = os.path.join(cache_dir, "papers")
            for p in tqdm(reranked_papers):
                if (not no_cache) and p.url:
                    cached = self._load_paper_cache(p.url, paper_cache_dir)
                    if p.tldr is None and isinstance(cached.get("tldr"), str) and cached["tldr"]:
                        p.tldr = cached["tldr"]
                    if p.affiliations is None and isinstance(cached.get("affiliations"), list):
                        p.affiliations = cached["affiliations"]

                if p.tldr is None:
                    p.generate_tldr(self._get_openai_client(), self.config.llm)
                    if (not no_cache) and p.url:
                        self._save_paper_cache(p.url, paper_cache_dir, tldr=p.tldr)

                if p.affiliations is None:
                    p.generate_affiliations(self._get_openai_client(), self.config.llm)
                    if (not no_cache) and p.url:
                        self._save_paper_cache(p.url, paper_cache_dir, affiliations=p.affiliations)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        email_content = render_email(reranked_papers)

        if not no_cache:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_html_path, "w", encoding="utf-8") as f:
                f.write(email_content)
            logger.info(f"HTML cache saved to {cache_html_path}")

        logger.info("Sending email...")
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
