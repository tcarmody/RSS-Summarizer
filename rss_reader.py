import os
import sys
import time
import json
import logging
import feedparser
import psutil
import anthropic
import requests
import traceback
import html
import re
from datetime import datetime
from functools import wraps
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dotenv import load_dotenv
from urllib.parse import urlparse
from sklearn.cluster import DBSCAN  # Import DBSCAN for clustering
import concurrent.futures
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry
import asyncio
from typing import List, Dict, Any, Optional
import torch
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

def create_session():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def performance_logger(func):
    """Decorator to log performance metrics of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        cpu_start = time.process_time()
        mem_start = psutil.Process().memory_info().rss

        result = func(*args, **kwargs)

        elapsed = time.time() - start_time
        cpu_percent = (time.process_time() - cpu_start) / elapsed * 100
        mem_used = psutil.Process().memory_info().rss - mem_start

        logging.info(f"Performance: {func.__name__} took {elapsed:.4f} seconds (CPU: {cpu_percent:.1f}%, Memory: {mem_used} bytes)")

        return result
    return wrapper

# Rate limit for Anthropic API (5 requests per second)
CALLS_PER_SECOND = 5
ANTHROPIC_RATE_LIMIT = limits(calls=CALLS_PER_SECOND, period=1)

@sleep_and_retry
@ANTHROPIC_RATE_LIMIT
def rate_limited_api_call(client: anthropic.Anthropic, messages: List[Dict[str, str]], **kwargs) -> Any:
    """Make a rate-limited call to the Anthropic API"""
    return client.messages.create(
        messages=messages,
        **kwargs
    )

class BatchProcessor:
    """Process items in batches with rate limiting."""

    def __init__(self, batch_size=5, requests_per_second=5):
        self.batch_size = batch_size
        self.delay = 1.0 / requests_per_second  # Time between requests
        self.queue = []
        self.results = []
        self.last_request_time = 0

    def add(self, item):
        """Add an item to the processing queue."""
        self.queue.append(item)
        if len(self.queue) >= self.batch_size:
            self._process_batch()

    def _process_batch(self):
        """Process a batch of items with rate limiting."""
        if not self.queue:
            return

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = []
            for item in self.queue[:self.batch_size]:
                # Ensure rate limiting
                now = time.time()
                time_since_last = now - self.last_request_time
                if time_since_last < self.delay:
                    time.sleep(self.delay - time_since_last)

                future = executor.submit(
                    item['func'],
                    *item['args'],
                    **item['kwargs']
                )
                futures.append(future)
                self.last_request_time = time.time()

            # Wait for all futures to complete
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                except Exception as e:
                    logging.error(f"Error in batch processing: {str(e)}")

        # Remove processed items
        self.queue = self.queue[self.batch_size:]

    def get_results(self):
        """Process remaining items and return all results."""
        while self.queue:
            self._process_batch()
        return self.results

"""
RSS Reader with AI-Powered Summarization and Article Clustering

A sophisticated RSS feed reader that combines AI-powered summarization using Claude 3,
efficient caching, and intelligent article clustering. Key features include:

- Claude 3 Haiku integration for high-quality article summarization
- Efficient caching system for feeds and summaries
- Semantic clustering of related articles using SentenceTransformer
- Export capabilities (HTML, PDF, Email)
- Readwise integration for saving articles
- Advanced content filtering (sponsored content, crypto)
- Smart rate limiting and batch processing
- Comprehensive logging and performance tracking

Updated: 2024-01-02
"""

import os
import re
import json
import time
import math
import logging
import psutil
import hashlib
import requests
import traceback
import html
from typing import Callable, Any, Dict, List, Optional, Union
import datetime
import pytz
import functools
import feedparser
import html2text
import sys
import newspaper
from newspaper import Article, Config
import webbrowser
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from urllib.parse import urlparse
from sklearn.cluster import DBSCAN  # Import DBSCAN for clustering
import concurrent.futures
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry
import asyncio
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

if not os.path.exists(os.path.join(os.path.dirname(__file__), '.env')):
    logging.warning(f"No .env file found at {os.path.join(os.path.dirname(__file__), '.env')}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rss_reader.log')
    ]
)

logger = logging.getLogger(__name__)

def get_env_var(var_name, required=True):
    """Get environment variable with error handling.

    Args:
        var_name (str): Name of environment variable
        required (bool): Whether the variable is required

    Returns:
        str: Value of environment variable

    Raises:
        ValueError: If required variable is not set
    """
    value = os.getenv(var_name)
    if required and not value:
        raise ValueError(
            f"Environment variable {var_name} is not set. "
            f"Please set it in your .env file."
        )
    return value

def track_performance(log_level=logging.INFO, log_to_file=True):
    """
    Decorator to track performance of methods with optional logging.

    Args:
        log_level (int): Logging level (default: logging.INFO)
        log_to_file (bool): Whether to log performance to a file (default: True)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Prepare performance tracking
            start_time = time.time()
            start_memory = os.getpid()

            # Track CPU and memory usage
            try:
                process = psutil.Process(start_memory)
                start_cpu_percent = process.cpu_percent()
                start_memory_info = process.memory_info()
            except ImportError:
                start_cpu_percent = None
                start_memory_info = None

            # Execute the function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # Log any exceptions during execution
                logging.error(f"Error in {func.__name__}: {e}")
                raise

            # Calculate performance metrics
            end_time = time.time()
            execution_time = end_time - start_time

            # Collect additional performance data
            try:
                if start_cpu_percent is not None:
                    end_cpu_percent = process.cpu_percent()
                    end_memory_info = process.memory_info()

                    cpu_usage = end_cpu_percent - start_cpu_percent
                    memory_usage = end_memory_info.rss - start_memory_info.rss
                else:
                    cpu_usage = None
                    memory_usage = None
            except Exception:
                cpu_usage = None
                memory_usage = None

            # Prepare performance log
            performance_log = {
                'function': func.__name__,
                'execution_time': execution_time,
                'timestamp': time.time(),
                'cpu_usage_percent': cpu_usage,
                'memory_usage_bytes': memory_usage,
                'args': str(args),
                'kwargs': str(kwargs)
            }

            # Logging
            log_message = (
                f"Performance: {func.__name__} "
                f"took {execution_time:.4f} seconds "
                f"(CPU: {cpu_usage}%, Memory: {memory_usage} bytes)"
            )

            # Log to console
            logging.log(log_level, log_message)

            # Optional file logging
            if log_to_file:
                # Ensure performance logs directory exists
                log_dir = os.path.join(os.path.dirname(__file__), 'performance_logs')
                os.makedirs(log_dir, exist_ok=True)

                # Create log filename with timestamp
                log_filename = os.path.join(
                    log_dir,
                    f"performance_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
                )

                # Append to performance log file
                try:
                    if os.path.exists(log_filename):
                        with open(log_filename, 'r+') as f:
                            try:
                                logs = json.load(f)
                            except json.JSONDecodeError:
                                logs = []
                            logs.append(performance_log)
                            f.seek(0)
                            json.dump(logs, f, indent=2)
                    else:
                        with open(log_filename, 'w') as f:
                            json.dump([performance_log], f, indent=2)
                except Exception as e:
                    logging.error(f"Could not write performance log: {e}")

            return result
        return wrapper
    return decorator


class SummaryCache:
    """
    A caching mechanism for article summaries to reduce redundant API calls
    and improve processing speed.
    """
    def __init__(self, cache_dir='.cache', cache_duration=7*24*60*60, max_cache_size=500):
        """
        Initialize the summary cache with configurable settings.

        Args:
            cache_dir (str): Directory to store cache files
            cache_duration (int): How long to keep summaries (in seconds)
            max_cache_size (int): Maximum number of entries in cache
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        self.max_cache_size = max_cache_size
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'summary_cache.json')
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load the cache from disk, creating an empty one if it doesn't exist."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Convert any string values to dict format
                    for key, value in data.items():
                        if isinstance(value, str):
                            self.cache[key] = {
                                'summary': value,
                                'timestamp': time.time()
                            }
                        else:
                            self.cache[key] = value
                # Clean up expired entries
                self._cleanup_cache()
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
            self.cache = {}

    def _save_cache(self):
        """Save the current cache to disk in JSON format."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    def _cleanup_cache(self):
        """Remove expired entries and enforce maximum cache size."""
        current_time = time.time()
        # Remove expired entries
        self.cache = {
            k: v for k, v in self.cache.items()
            if isinstance(v, dict) and current_time - v.get('timestamp', 0) < self.cache_duration
        }

        # If still too large, remove oldest entries
        if len(self.cache) > self.max_cache_size:
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].get('timestamp', 0) if isinstance(x[1], dict) else 0
            )
            self.cache = dict(sorted_items[-self.max_cache_size:])

    def get(self, text):
        """Retrieve cached summary for a given text."""
        key = self._hash_text(text)
        if key in self.cache:
            entry = self.cache[key]
            if isinstance(entry, dict) and time.time() - entry.get('timestamp', 0) < self.cache_duration:
                return entry
            else:
                del self.cache[key]
        return None

    def set(self, text, summary):
        """Cache a summary for a given text."""
        key = self._hash_text(text)
        if isinstance(summary, str):
            summary = {'summary': summary}
        summary['timestamp'] = time.time()
        self.cache[key] = summary
        if len(self.cache) > self.max_cache_size:
            self._cleanup_cache()
        self._save_cache()

    def _hash_text(self, text):
        """Generate a hash for the given text to use as a cache key."""
        # Convert text to string if it's not already
        if not isinstance(text, str):
            text = str(text)
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def clear_cache(self):
        """Completely clear the cache."""
        self.cache = {}
        try:
            os.remove(self.cache_file)
        except FileNotFoundError:
            pass
        self._save_cache()


class ArticleSummarizer:
    """Summarizes articles using Claude API."""

    def __init__(self):
        """Initialize the summarizer with Claude API client."""
        self.client = anthropic.Anthropic(api_key=get_env_var('ANTHROPIC_API_KEY'))
        self.summary_cache = SummaryCache()

    def clean_text(self, text):
        """Clean HTML and normalize text for summarization."""
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Decode HTML entities
        text = html.unescape(text)

        return text

    def summarize_article(self, text, force_refresh=False):
        """Generate a concise summary of the article text."""
        # Clean the text first
        text = self.clean_text(text)

        # Check cache first
        cached_summary = self.summary_cache.get(text)
        if cached_summary and not force_refresh:
            return cached_summary['summary']

        try:
            # Generate summary using Claude
            prompt = (
                "Summarize this article in 3-4 sentences using active voice and factual tone. "
                "Follow this structure:\n"
                "1. First line: Create a headline in sentence case\n"
                "2. Then a blank line\n"
                "3. Then the summary that:\n"
                "- Explains what happened in simple language\n"
                "- Identifies key details for AI developers\n"
                "- Explains why it matters to AI industry followers\n"
                "- Spells out numbers and uses U.S./U.K. with periods\n\n"
                f"Article:\n{text}\n\n"
                f"URL: {article['link']}"
            )

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=400,
                temperature=0.3,
                system="You are an expert AI technology journalist. Be concise and factual.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            summary_text = response.content[0].text

            # Split into headline and summary
            lines = summary_text.split('\n', 1)
            if len(lines) == 2:
                headline = lines[0].strip()
                summary = lines[1].strip()
            else:
                headline = article['title']
                summary = summary_text

            return {
                'headline': headline,
                'summary': summary
            }

        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return {
                'headline': article['title'],
                'summary': "Summary generation failed. Please try again later."
            }

    def generate_tags(self, content):
        """Generate tags for an article using Claude."""
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0.7,
                system="Extract specific entities from the text and return them as tags. Include:\n- Company names (e.g., 'Apple', 'Microsoft')\n- Technologies (e.g., 'ChatGPT', 'iOS 17')\n- People (e.g., 'Tim Cook', 'Satya Nadella')\n- Products (e.g., 'iPhone 15', 'Surface Pro')\nFormat: Return only the tags as a comma-separated list, with no categories or explanations.",
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            tags = [tag.strip() for tag in response.content[0].text.split(',')]
            return tags
        except Exception as e:
            logging.error(f"Error generating tags: {str(e)}")
            return []

class RSSReader:
    """
    Main class that handles RSS feed processing, article summarization, and clustering.

    This class orchestrates the entire process of:
    1. Fetching and parsing RSS feeds
    2. Generating AI-powered summaries
    3. Clustering similar articles
    4. Generating HTML output

    The class uses Claude API for high-quality summaries and semantic similarity
    for clustering related articles. It implements caching to avoid redundant
    API calls and includes fallback options for summarization.
    """

    def __init__(self, feeds=None, batch_size=25, batch_delay=15):
        """Initialize RSSReader with feeds and settings."""
        self.feeds = feeds or self._load_default_feeds()
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.session = create_session()
        self.client = anthropic.Anthropic()
        self.batch_processor = BatchProcessor(batch_size=5)  # Process 5 API calls at a time

        # Initialize sentence transformer and device
        self.model = None
        self.device = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence transformer model and device."""
        try:
            if self.model is None:
                logging.info("Initializing sentence transformer model...")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = SentenceTransformer('all-mpnet-base-v2')
                self.model = self.model.to(self.device)
                logging.info(f"Model initialized on device: {self.device}")
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    def _load_default_feeds(self):
        """Load feed URLs from the default file."""
        feeds = []
        try:
            with open('rss_test.txt', 'r') as f:
                for line in f:
                    url = line.strip()
                    if url and not url.startswith('#'):
                        # Remove any inline comments and clean the URL
                        url = url.split('#')[0].strip()
                        url = ''.join(c for c in url if ord(c) >= 32)  # Remove control characters
                        if url:
                            feeds.append(url)
            return feeds
        except Exception as e:
            logging.error(f"Error loading feed URLs: {str(e)}")
            return []

    def process_cluster_summaries(self, clusters):
        processed_clusters = []
        for i, cluster in enumerate(clusters, 1):
            try:
                if not cluster:
                    logging.warning(f"Empty cluster {i}, skipping")
                    continue

                logging.info(f"Processing cluster {i}/{len(clusters)} with {len(cluster)} articles")

                if len(cluster) > 1:
                    # For clusters with multiple articles, generate a combined summary
                    combined_text = "\n\n".join([
                        f"Title: {article['title']}\n{article.get('content', '')[:1000]}"
                        for article in cluster
                    ])

                    logging.info(f"Generating summary for cluster {i} with {len(cluster)} articles")
                    cluster_summary = self._generate_summary(combined_text,
                                                            f"Combined summary of {len(cluster)} related articles",
                                                            cluster[0]['link'])

                    # Add the cluster summary to each article
                    for article in cluster:
                        article['summary'] = cluster_summary
                        article['cluster_size'] = len(cluster)
                else:
                    # Single article
                    article = cluster[0]
                    if not article.get('summary'):
                        logging.info(f"Generating summary for single article: {article['title']}")
                        article['summary'] = self._generate_summary(
                            article.get('content', ''),
                            article['title'],
                            article['link']
                        )
                    article['cluster_size'] = 1

                processed_clusters.append(cluster)
                logging.info(f"Successfully processed cluster {i}")

            except Exception as cluster_error:
                logging.error(f"Error processing cluster {i}: {str(cluster_error)}", exc_info=True)
                continue

        return processed_clusters

    def _parse_entry(self, entry, feed_title):
        """Parse a feed entry into an article."""
        try:
            # Extract content
            content = ''
            if hasattr(entry, 'content'):
                raw_content = entry.content
                if isinstance(raw_content, list) and raw_content:
                    content = raw_content[0].get('value', '')
                elif isinstance(raw_content, str):
                    content = raw_content
                elif isinstance(raw_content, (list, tuple)):
                    content = ' '.join(str(item) for item in raw_content)
                else:
                    content = str(raw_content)

            # Fallback to summary
            if not content and hasattr(entry, 'summary'):
                content = entry.summary

            # Final fallback to title
            if not content:
                content = getattr(entry, 'title', '')
                logging.warning("Using title as content fallback")

            # Clean content
            content = html.unescape(content)
            content = re.sub(r'<[^>]+>', '', content)
            content = content.strip()

            return {
                'title': getattr(entry, 'title', 'No Title'),
                'link': getattr(entry, 'link', '#'),
                'published': getattr(entry, 'published', 'Unknown date'),
                'content': content,
                'feed_source': feed_title
            }

        except Exception as e:
            logging.error(f"Error parsing entry: {str(e)}")
            return None

    @track_performance()
    def process_feeds(self):
        """Process all RSS feeds and generate summaries."""
        try:
            all_articles = []

            # Process feeds in batches
            for batch in self._get_feed_batches():
                logging.info(f"\n🔄 Processing Batch {batch['current']}/{batch['total']}: "
                           f"Feeds {batch['start']} to {batch['end']}")

                # Process each feed in the batch in parallel
                with ThreadPoolExecutor(max_workers=min(len(batch['feeds']), 10)) as executor:
                    futures = [executor.submit(self._process_feed, feed) for feed in batch['feeds']]
                    batch_articles = []
                    for future in as_completed(futures):
                        articles = future.result()
                        if articles:
                            batch_articles.extend(articles)
                            logging.info(f"Added {len(articles)} articles to batch")

                all_articles.extend(batch_articles)
                logging.info(f"Batch complete. Total articles so far: {len(all_articles)}")

                # Add delay between batches if there are more
                if batch['current'] < batch['total']:
                    time.sleep(self.batch_delay)

            logging.info(f"Total articles collected: {len(all_articles)}")

            if not all_articles:
                logging.error("No articles collected from any feeds")
                return None

            # First cluster the articles
            logging.info("Clustering similar articles...")
            clusters = self._cluster_articles(all_articles)

            if not clusters:
                logging.error("No clusters created")
                return None

            logging.info(f"Created {len(clusters)} clusters")

            # Now generate summaries for each cluster
            logging.info("Generating summaries for article clusters...")
            processed_clusters = self.process_cluster_summaries(clusters)

            if not processed_clusters:
                logging.error("No clusters were successfully processed")
                return None

            logging.info(f"Successfully processed {len(processed_clusters)} clusters")

            # Generate HTML output
            output_file = self.generate_html_output(processed_clusters)
            if output_file:
                logging.info(f"Successfully generated HTML output: {output_file}")
            else:
                logging.error("Failed to generate HTML output")

            return output_file

        except Exception as e:
            logging.error(f"Error processing feeds: {str(e)}", exc_info=True)
            return None

    def _process_feed(self, feed_url):
        """Process a single RSS feed."""
        try:
            feed = feedparser.parse(feed_url)
            articles = []

            if feed.entries:
                feed_title = feed.feed.get('title', feed_url)
                logging.info(f"📰 Found {len(feed.entries)} articles in feed: {feed_url}")

                for entry in feed.entries[:self.batch_size]:
                    article = self._parse_entry(entry, feed_title)
                    if article:
                        articles.append(article)

            return articles

        except Exception as e:
            logging.error(f"Error processing feed {feed_url}: {str(e)}")
            return []

    def _generate_summary(self, article_text, title, url):
        """Generate a summary for an article using the Anthropic API."""
        try:
            # Generate summary using Claude
            prompt = (
                "Summarize this article in 3-4 sentences using active voice and factual tone. "
                "Follow this structure:\n"
                "1. First line: Create a descriptive title that captures the key theme or insight. The title should:\n"
                "   - Be 5-10 words long\n"
                "   - Use sentence case\n"
                "   - Focus on the main technological development, business impact, or key finding\n"
                "   - Be specific rather than generic (e.g. 'OpenAI launches GPT-4 with enhanced reasoning' rather than 'AI company releases new model')\n"
                "2. Then a blank line\n"
                "3. Then the summary that:\n"
                "- Explains what happened in simple language\n"
                "- Identifies key details for AI developers\n"
                "- Explains why it matters to AI industry followers\n"
                "- Spells out numbers and uses U.S./U.K. with periods\n\n"
                f"Article:\n{article_text}\n\n"
                f"URL: {url}"
            )

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=400,
                temperature=0.3,
                system="You are an expert AI technology journalist. Be concise and factual.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            summary_text = response.content[0].text

            # Split into headline and summary
            lines = summary_text.split('\n', 1)
            if len(lines) == 2:
                headline = lines[0].strip()
                summary = lines[1].strip()
            else:
                headline = title
                summary = summary_text

            return {
                'headline': headline,
                'summary': summary
            }

        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return {
                'headline': title,
                'summary': "Summary generation failed. Please try again later."
            }

    def _cluster_articles(self, articles):
        """Cluster similar articles together using sentence embeddings."""
        try:
            # Log performance metrics
            start_time = time.time()

            if not articles:
                logging.warning("No articles to cluster")
                return []

            logging.info(f"Clustering {len(articles)} articles")

            # Initialize model if needed
            if self.model is None:
                self._initialize_model()

            # Combine title and first part of content for better context
            texts = []
            for article in articles:
                title = article.get('title', '')
                content = article.get('content', '')[:500]  # First 500 chars of content
                combined_text = f"{title} {content}".strip()
                texts.append(combined_text)
                logging.debug(f"Processing article for clustering: {title}")

            # Get embeddings with progress bar
            logging.info("Generating embeddings for articles...")
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                normalize_embeddings=True
            )

            # Use Agglomerative Clustering
            logging.info("Clustering articles...")
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,  # Adjust based on testing
                metric='cosine',
                linkage='average'
            ).fit(embeddings)

            # Group articles by cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(clustering.labels_):
                clusters[label].append(articles[idx])

            # Convert to list and sort by size
            clustered_articles = list(clusters.values())
            clustered_articles.sort(key=len, reverse=True)

            # Log clustering results
            logging.info(f"Created {len(clustered_articles)} clusters:")
            for i, cluster in enumerate(clustered_articles):
                titles = [a.get('title', 'No title') for a in cluster]
                logging.info(f"Cluster {i}: {len(cluster)} articles")
                logging.info(f"Titles: {titles}")

            return clustered_articles

        except Exception as e:
            logging.error(f"Error clustering articles: {str(e)}", exc_info=True)
            # Fallback: return each article in its own cluster
            return [[article] for article in articles]

    def _get_feed_batches(self):
        """Generate batches of feeds to process."""
        logging.info("🚀 Initializing RSS Reader...")
        logging.info(f"📊 Total Feeds: {len(self.feeds)}")
        logging.info(f"📦 Batch Size: {self.batch_size}")
        logging.info(f"⏱️  Batch Delay: {self.batch_delay} seconds")

        total_batches = (len(self.feeds) + self.batch_size - 1) // self.batch_size
        logging.info(f"🔄 Total Batches: {total_batches}")

        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, len(self.feeds))
            yield {
                'current': batch_num + 1,
                'total': total_batches,
                'start': start_idx + 1,
                'end': end_idx,
                'feeds': self.feeds[start_idx:end_idx]
            }

    def generate_html_output(self, clusters):
        """Generate HTML output from the processed clusters."""
        try:
            from flask import Flask, render_template
            from datetime import datetime
            import os

            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(__file__), 'output')
            os.makedirs(output_dir, exist_ok=True)

            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'rss_summary_{timestamp}.html')

            app = Flask(__name__)

            with app.app_context():
                html_content = render_template(
                    'feed_summary.html',
                    clusters=clusters,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                logging.info(f"Successfully wrote HTML output to {output_file}")
                return output_file

        except Exception as e:
            logging.error(f"Error generating HTML output: {str(e)}", exc_info=True)
            return False


def main():
    """
    Main function to run the RSS reader.
    """
    try:
        # Initialize and run RSS reader
        rss_reader = RSSReader()
        output_file = rss_reader.process_feeds()

        if not output_file:
            logging.warning("⚠️ No articles found or processed")

    except Exception as e:
        logging.error(f"❌ Error in main: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
