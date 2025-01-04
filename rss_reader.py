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
from sentence_transformers import SentenceTransformer
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

# Load environment variables from .env file
env_path = os.path.expanduser('~/workspace/.env')
load_dotenv(env_path)

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
env_path = os.path.expanduser('~/workspace/.env')
load_dotenv(env_path)

if not os.path.exists(env_path):
    logging.warning(f"No .env file found at {env_path}")

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

class FeedCache:
    """
    A caching mechanism for RSS feed entries to reduce redundant network requests
    and improve processing speed.
    """
    def __init__(self, 
                 cache_dir='/tmp/rss_reader_cache', 
                 cache_duration=timedelta(hours=1),
                 max_cache_size=500):
        """
        Initialize the feed cache.
        
        :param cache_dir: Directory to store cached feed entries
        :param cache_duration: How long to keep cache entries valid
        :param max_cache_size: Maximum number of cache entries to keep
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        self.max_cache_size = max_cache_size
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, url):
        """Generate a unique cache key for a given URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _prune_cache(self):
        """Remove old cache entries if cache exceeds max size."""
        cache_files = [
            os.path.join(self.cache_dir, f) 
            for f in os.listdir(self.cache_dir)
        ]
        
        # Sort files by modification time
        cache_files.sort(key=lambda x: os.path.getmtime(x))
        
        # Remove oldest files if over max cache size
        while len(cache_files) > self.max_cache_size:
            # Remove oldest files first
            oldest_file = cache_files.pop(0)
            try:
                os.remove(oldest_file)
            except Exception as e:
                print(f"Error removing cache file {oldest_file}: {e}")
    
    def get(self, url):
        """
        Retrieve cached content for a given URL.
        
        :param url: URL to retrieve cache for
        :return: Cached content or None if not found/expired
        """
        cache_path = os.path.join(self.cache_dir, self._get_cache_key(url))
        
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid
                cache_timestamp = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cache_timestamp < self.cache_duration:
                    return cached_data['content']
        except Exception as e:
            print(f"Error reading cache for {url}: {e}")
        
        return None
    
    def set(self, url, content):
        """
        Store content in cache for a given URL.
        
        :param url: URL to cache content for
        :param content: Content to cache
        """
        try:
            # Prune cache if needed
            self._prune_cache()
            
            # Create cache file
            cache_path = os.path.join(self.cache_dir, self._get_cache_key(url))
            with open(cache_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'content': content
                }, f)
        except Exception as e:
            print(f"Error writing cache for {url}: {e}")

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

class RSSReaderConfig:
    """Configuration class for RSS Reader settings."""
    def __init__(self):
        # Filtering settings
        self.min_article_words = 50
        self.max_article_words = 5000
        self.keywords = ['technology', 'ai', 'innovation', 'science']
        self.exclude_keywords = ['advertisement', 'sponsored', 'clickbait']
        
        # Rate limiting settings
        self.max_requests_per_minute = 30
        
        # Logging settings
        self.log_level = logging.INFO
        self.log_file = 'rss_reader.log'

class RateLimiter:
    """
    A decorator to limit the rate of function calls.
    
    Prevents overwhelming external services by controlling 
    the frequency of requests.
    """
    def __init__(self, max_calls_per_minute):
        """
        Initialize rate limiter.
        
        :param max_calls_per_minute: Maximum number of calls allowed per minute
        """
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Remove old calls outside the time window
            current_time = time.time()
            self.calls = [
                call_time for call_time in self.calls 
                if current_time - call_time < 60
            ]
            
            # Check if we've exceeded rate limit
            if len(self.calls) >= self.max_calls_per_minute:
                # Calculate time to wait
                oldest_call = self.calls[0]
                wait_time = 60 - (current_time - oldest_call)
                if wait_time > 0:
                    logging.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
                    time.sleep(wait_time)
            
            # Record this call and execute the function
            self.calls.append(time.time())
            return func(*args, **kwargs)
        
        return wrapper

class ArticleFilter:
    """Advanced content filtering system for RSS feed articles.
    
    Provides flexible filtering based on multiple criteria including:
    - Sponsored/advertising content detection
    - Cryptocurrency content filtering
    - Custom keyword and pattern matching
    """
    
    def __init__(self):
        # Patterns that indicate sponsored/advertising content
        self.sponsored_patterns = [
            r'(?i)sponsored\s+(?:by|content|post)',
            r'(?i)advertisement\s*(?:feature)?',
            r'(?i)promoted\s+(?:by|content|post)',
            r'(?i)paid\s+(?:content|post|partnership)',
            r'(?i)partner\s+content',
            r'(?i)\[sponsored\]',
            r'(?i)in\s+association\s+with'
        ]
        
        # Patterns for crypto-related content
        self.crypto_patterns = [
            r'(?i)(?:bit|lite|doge)coin',
            r'(?i)crypto(?:currency|currencies)',
            r'(?i)blockchain',
            r'(?i)(?:web3|nft)',
            r'(?i)(?:defi|dao|dex)',
            r'(?i)(?:mining rig|hash rate)',
            r'(?i)(?:hodl|to the moon)',
            r'(?i)(?:satoshi|nakamoto)'
        ]
        
        # Threshold for crypto content (percentage of content that can be crypto-related)
        self.crypto_threshold = 0.15  # 15% threshold
        
    def is_sponsored_content(self, article):
        """Check if article is sponsored content."""
        # Check title, description, and content
        text_to_check = ' '.join(filter(None, [
            article.get('title', ''),
            article.get('description', ''),
            article.get('content', '')
        ])).lower()
        
        # Check for sponsored patterns
        return bool(self.sponsored_regex.search(text_to_check))
    
    def is_crypto_focused(self, article):
        """Check if article is primarily about cryptocurrency."""
        text_to_check = ' '.join(filter(None, [
            article.get('title', ''),
            article.get('description', ''),
            article.get('content', '')
        ])).lower()
        
        # Split into words for analysis
        words = text_to_check.split()
        if not words:
            return False
            
        # Count crypto-related matches
        crypto_matches = len(re.findall(self.crypto_regex, text_to_check))
        
        # Calculate the ratio of crypto-related terms to total words
        crypto_ratio = crypto_matches / len(words)
        
        return crypto_ratio > self.crypto_threshold
    
    def filter_articles(self, articles, config=None):
        """Filter articles based on predefined criteria.
        
        Args:
            articles: List of article dictionaries
            config: Optional configuration settings
            
        Returns:
            List of filtered articles
        """
        filtered_articles = []
        
        for article in articles:
            # Skip if article is missing essential fields
            if not all(k in article for k in ['title', 'link']):
                continue
                
            # Skip sponsored content
            if self.is_sponsored_content(article):
                logging.debug(f"Filtered sponsored content: {article['title']}")
                continue
                
            # Skip crypto-focused content
            if self.is_crypto_focused(article):
                logging.debug(f"Filtered crypto content: {article['title']}")
                continue
                
            # Apply additional config-based filters if provided
            if config:
                # Check minimum word count
                if hasattr(config, 'min_article_words'):
                    content = article.get('content', '')
                    word_count = len(content.split())
                    if word_count < config.min_article_words:
                        continue
                
                # Check excluded keywords
                if hasattr(config, 'exclude_keywords'):
                    text = ' '.join([
                        article.get('title', ''),
                        article.get('description', ''),
                        article.get('content', '')
                    ]).lower()
                    
                    if any(keyword.lower() in text for keyword in config.exclude_keywords):
                        continue
            
            filtered_articles.append(article)
            
        return filtered_articles

class EnhancedLogging:
    """
    Enhanced logging mechanism for the RSS reader.
    Provides more detailed and structured logging.
    """
    @staticmethod
    def setup_logging(config: RSSReaderConfig):
        """
        Configure logging with specified settings.
        
        :param config: Configuration settings
        """
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(config.log_file) or '.'
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.log_file or 'rss_reader.log'),
                logging.StreamHandler()  # Also log to console
            ]
        )
        
        # Create a logger for the RSS reader
        logger = logging.getLogger('RSSReader')
        return logger

class TagGenerator:
    """Generates and suggests tags based on article content analysis."""
    
    def __init__(self):
        # General topic categories
        self.topic_patterns = {
            'technology': [
                r'(?i)(artificial intelligence|machine learning|AI|ML|deep learning)',
                r'(?i)(software|programming|coding|development)',
                r'(?i)(cloud computing|aws|azure|google cloud)',
                r'(?i)(cybersecurity|security|privacy|encryption)',
                r'(?i)(mobile|ios|android|app)'
            ],
            'business': [
                r'(?i)(startup|venture capital|vc funding|investment)',
                r'(?i)(market|stock|shares|investor|trading)',
                r'(?i)(revenue|profit|earnings|quarterly|fiscal)',
                r'(?i)(merger|acquisition|partnership|deal)',
                r'(?i)(ceo|executive|leadership|management)'
            ],
            'research': [
                r'(?i)(study|research|paper|publication)',
                r'(?i)(scientists|researchers|professors)',
                r'(?i)(university|institute|laboratory)',
                r'(?i)(experiment|discovery|breakthrough)',
                r'(?i)(journal|peer-review|methodology)'
            ],
            'policy': [
                r'(?i)(regulation|policy|legislation|law)',
                r'(?i)(government|federal|state|agency)',
                r'(?i)(compliance|guidelines|framework)',
                r'(?i)(senate|congress|bill|act)',
                r'(?i)(election|political|campaign|vote)'
            ]
        }
        
        # Company detection patterns
        self.company_indicators = [
            r'(?i)(Inc\.|Corp\.|LLC|Ltd\.)',
            r'(?i)(Company|Corporation)',
            r'(?i)(\w+\.com)',
            r'(?i)announced|released|launched|unveiled'
        ]
        
        # Compile patterns for better performance
        self.compiled_patterns = {
            category: [re.compile(pattern) for pattern in patterns]
            for category, patterns in self.topic_patterns.items()
        }
        self.compiled_company_patterns = [re.compile(pattern) 
                                        for pattern in self.company_indicators]
        
        # Load spaCy model for entity recognition
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logging.warning("spaCy model not found. Installing now...")
            os.system('python -m spacy download en_core_web_sm')
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
    
    def extract_companies(self, text):
        """Extract company names from text using NER and pattern matching."""
        companies = set()
        
        # Use spaCy for entity recognition
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['ORG']:
                companies.add(ent.text.strip())
        
        # Pattern-based company detection
        sentences = text.split('.')
        for sentence in sentences:
            for pattern in self.compiled_company_patterns:
                if pattern.search(sentence):
                    # Look for capitalized words before company indicators
                    words = sentence.split()
                    for i, word in enumerate(words):
                        if word[0].isupper() and i > 0 and words[i-1][0].isupper():
                            company = ' '.join(words[i-1:i+1])
                            companies.add(company.strip())
        
        return list(companies)
    
    def analyze_content_type(self, text):
        """Determine if content is news, analysis, opinion, or tutorial."""
        indicators = {
            'news': [
                r'(?i)(today|yesterday|announced|released)',
                r'(?i)(breaking|latest|update|report)',
                r'(?i)(according to|sources say)'
            ],
            'analysis': [
                r'(?i)(analysis|examine|investigate)',
                r'(?i)(implications|impact|effects)',
                r'(?i)(trend|pattern|development)'
            ],
            'opinion': [
                r'(?i)(opinion|viewpoint|perspective)',
                r'(?i)(argue|believe|think|suggest)',
                r'(?i)(should|must|need to)'
            ],
            'tutorial': [
                r'(?i)(how to|guide|tutorial)',
                r'(?i)(step by step|learn|implement)',
                r'(?i)(example|sample|code)'
            ]
        }
        
        scores = {category: 0 for category in indicators}
        for category, patterns in indicators.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text))
                scores[category] += matches
        
        # Return the category with highest score
        if any(scores.values()):
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def generate_tags(self, article):
        """Generate tags based on article content analysis.
        
        Args:
            article (dict): Article dictionary containing title and content
            
        Returns:
            dict: Dictionary containing suggested tags in different categories
        """
        text = f"{article.get('title', '')} {article.get('content', '')}"
        
        # Initialize tags
        tags = {
            'topics': set(),
            'companies': set(),
            'content_type': set(),
            'general': set()
        }
        
        # Extract topics
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    tags['topics'].add(category)
        
        # Extract companies
        companies = self.extract_companies(text)
        tags['companies'].update(companies)
        
        # Determine content type
        content_type = self.analyze_content_type(text)
        tags['content_type'].add(content_type)
        
        # Extract key phrases using spaCy
        doc = self.nlp(text[:10000])  # Limit text length for performance
        for chunk in doc.noun_chunks:
            # Only add noun phrases that are likely to be meaningful
            if (len(chunk.text.split()) > 1 and  # Multi-word phrases
                chunk.root.pos_ in ['NOUN', 'PROPN'] and  # Noun-based
                not any(char.isdigit() for char in chunk.text)):  # No numbers
                tags['general'].add(chunk.text.strip().lower())
        
        # Convert sets to sorted lists
        return {
            category: sorted(list(tags))
            for category, tags in tags.items()
        }

class FavoritesManager:
    """Manages saved and favorite articles with persistence and organization features."""
    
    def __init__(self, storage_dir='.cache'):
        """Initialize the favorites manager.
        
        Args:
            storage_dir (str): Directory to store favorites data
        """
        self.storage_dir = storage_dir
        self.favorites_file = os.path.join(storage_dir, 'favorites.json')
        self.favorites = self._load_favorites()
        self.tag_generator = TagGenerator()
        self.export_manager = ExportManager()
        self.readwise = None  # Lazy initialization
    
    def _load_favorites(self):
        """Load favorites from disk."""
        try:
            if os.path.exists(self.favorites_file):
                with open(self.favorites_file, 'r') as f:
                    return json.load(f)
            return {
                'articles': [],
                'tags': {},
                'last_updated': None
            }
        except Exception as e:
            logging.error(f"Error loading favorites: {e}")
            return {'articles': [], 'tags': {}, 'last_updated': None}
    
    def _save_favorites(self):
        """Save favorites to disk."""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.favorites_file, 'w') as f:
                self.favorites['last_updated'] = datetime.now().isoformat()
                json.dump(self.favorites, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving favorites: {e}")
    
    def add_favorite(self, article, tags=None):
        """Add an article to favorites with suggested tags."""
        try:
            # Generate suggested tags if none provided
            if not tags:
                suggested_tags = self.tag_generator.generate_tags(article)
                # Flatten suggested tags into a single list
                tags = []
                for category, category_tags in suggested_tags.items():
                    if category == 'companies':
                        tags.extend([f"company:{tag}" for tag in category_tags])
                    elif category == 'content_type':
                        tags.extend([f"type:{tag}" for tag in category_tags])
                    else:
                        tags.extend(category_tags)
            
            # Create a unique ID for the article
            article_id = hashlib.md5(article['link'].encode()).hexdigest()
            
            # Prepare article data with suggested tags
            favorite = {
                'id': article_id,
                'title': article.get('title', ''),
                'link': article['link'],
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'summary': article.get('summary', ''),
                'pub_date': article.get('pub_date', ''),
                'added_date': datetime.now().isoformat(),
                'tags': tags,
                'suggested_tags': suggested_tags,  # Store full tag analysis
                'notes': ''
            }
            
            # Update or add article
            existing = next((a for a in self.favorites['articles'] 
                           if a['id'] == article_id), None)
            if existing:
                existing.update(favorite)
            else:
                self.favorites['articles'].append(favorite)
            
            # Update tag index
            for tag in tags:
                if tag not in self.favorites['tags']:
                    self.favorites['tags'][tag] = []
                if article_id not in self.favorites['tags'][tag]:
                    self.favorites['tags'][tag].append(article_id)
            
            self._save_favorites()
            return True
            
        except Exception as e:
            logging.error(f"Error adding favorite: {e}")
            return False
    
    def remove_favorite(self, article_id):
        """Remove an article from favorites."""
        try:
            # Remove from articles list
            self.favorites['articles'] = [
                a for a in self.favorites['articles'] 
                if a['id'] != article_id
            ]
            
            # Remove from tags
            for tag_list in self.favorites['tags'].values():
                if article_id in tag_list:
                    tag_list.remove(article_id)
            
            self._save_favorites()
            return True
        except Exception as e:
            logging.error(f"Error removing favorite: {e}")
            return False
    
    def add_tags(self, article_id, tags):
        """Add tags to an article."""
        article = next((a for a in self.favorites['articles'] 
                       if a['id'] == article_id), None)
        if article:
            article['tags'] = list(set(article['tags'] + tags))
            for tag in tags:
                if tag not in self.favorites['tags']:
                    self.favorites['tags'][tag] = []
                if article_id not in self.favorites['tags'][tag]:
                    self.favorites['tags'][tag].append(article_id)
            self._save_favorites()
            return True
        return False
    
    def remove_tag(self, article_id, tag):
        """Remove a tag from an article."""
        article = next((a for a in self.favorites['articles'] 
                       if a['id'] == article_id), None)
        if article and tag in article['tags']:
            article['tags'].remove(tag)
            if tag in self.favorites['tags']:
                self.favorites['tags'][tag].remove(article_id)
            self._save_favorites()
            return True
        return False
    
    def add_note(self, article_id, note):
        """Add a note to a favorite article."""
        article = next((a for a in self.favorites['articles'] 
                       if a['id'] == article_id), None)
        if article:
            article['notes'] = note
            self._save_favorites()
            return True
        return False
    
    def get_favorites(self, tag=None):
        """Get all favorite articles, optionally filtered by tag."""
        if tag:
            if tag in self.favorites['tags']:
                article_ids = self.favorites['tags'][tag]
                return [a for a in self.favorites['articles'] 
                       if a['id'] in article_ids]
            return []
        return self.favorites['articles']
    
    def get_tags(self):
        """Get all tags and their counts."""
        return {tag: len(articles) for tag, articles 
                in self.favorites['tags'].items()}
    
    def search_favorites(self, query):
        """Search favorites by title, content, or tags."""
        query = query.lower()
        results = []
        
        for article in self.favorites['articles']:
            if (query in article['title'].lower() or
                query in article['description'].lower() or
                query in article['content'].lower() or
                query in article['notes'].lower() or
                any(query in tag.lower() for tag in article['tags'])):
                results.append(article)
        
        return results
    
    def export_favorites(self, output_file=None, format='html', tag=None, 
                        include_summary=True):
        """Export favorites to various formats.
        
        Args:
            output_file (str): Output file path
            format (str): Export format ('html', 'email', 'pdf', or 'readwise')
            tag (str): Optional tag to filter articles
            include_summary (bool): Whether to include article summaries
        
        Returns:
            str: Path to output file or response from export
        """
        # Get articles to export
        articles = self.get_favorites(tag)
        
        if not articles:
            logging.warning("No articles to export")
            return None
            
        # Generate title
        title = "Favorite Articles"
        if tag:
            title += f" - {tag}"
            
        if format == 'pdf':
            return self.export_manager.generate_pdf(
                articles, title, output_file, include_summary
            )
        elif format == 'readwise':
            try:
                readwise = self._get_readwise()
                results = []
                for article in articles:
                    result = readwise.save_article(article)
                    results.append(result)
                return results
            except Exception as e:
                logging.error(f"Error exporting to Readwise: {e}")
                raise
        else:
            # Handle existing formats (html, email)
            return self.export_manager.export_favorites(output_file, format, tag, include_summary)
    
    def export_article(self, article_id, format='html', include_summary=True):
        """Export a single article.
        
        Args:
            article_id (str): ID of article to export
            format (str): Export format ('html', 'email', 'pdf', or 'readwise')
            include_summary (bool): Whether to include article summary
        """
        article = next((a for a in self.favorites['articles'] 
                       if a['id'] == article_id), None)
        
        if not article:
            logging.warning(f"Article not found: {article_id}")
            return None
        
        if format == 'pdf':
            return self.export_manager.generate_pdf(
                [article],
                article['title'],
                include_summary=include_summary
            )
        elif format == 'readwise':
            try:
                readwise = self._get_readwise()
                return readwise.save_article(article)
            except Exception as e:
                logging.error(f"Error exporting to Readwise: {e}")
                raise
        else:
            # Handle existing formats (html, email)
            return self.export_manager.export_article(article_id, format, include_summary)
    
    def _get_readwise(self):
        """Lazy initialization of Readwise client."""
        if not self.readwise:
            self.readwise = ReadwiseClient()
        return self.readwise

class ExportManager:
    """Handles exporting of articles and favorites to various formats."""
    
    def __init__(self):
        self.html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{title}</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }
                .article {
                    margin-bottom: 30px;
                    padding: 20px;
                    border: 1px solid #eee;
                    border-radius: 5px;
                    background: #fff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .article h2 {
                    margin-top: 0;
                    color: #2c5282;
                }
                .article a {
                    color: #2b6cb0;
                    text-decoration: none;
                }
                .article a:hover {
                    text-decoration: underline;
                }
                .meta {
                    font-size: 0.9em;
                    color: #666;
                    margin-bottom: 10px;
                }
                .tags {
                    margin-top: 10px;
                }
                .tag {
                    display: inline-block;
                    padding: 2px 8px;
                    margin: 2px;
                    background: #edf2f7;
                    border-radius: 12px;
                    font-size: 0.85em;
                    color: #4a5568;
                }
                .notes {
                    margin-top: 10px;
                    padding: 10px;
                    background: #fffaf0;
                    border-left: 3px solid #ed8936;
                }
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {content}
        </body>
        </html>
        '''
        
        self.email_template = '''
        <html>
        <body style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <h1 style="color: #2c5282;">{title}</h1>
            {content}
        </body>
        </html>
        '''
        
        # PDF-specific CSS additions
        self.pdf_css = '''
            @page {
                margin: 2.5cm;
                @top-center {
                    content: string(title);
                }
                @bottom-center {
                    content: counter(page);
                }
            }
            
            body {
                font-size: 11pt;
            }
            
            h1 {
                string-set: title content();
                page-break-before: always;
            }
            
            .article {
                page-break-inside: avoid;
            }
            
            @media print {
                a {
                    color: #000;
                    text-decoration: none;
                }
                
                .article {
                    border: none;
                    box-shadow: none;
                }
            }
        '''
        
        self.summary_css = '''
            .summary-box {
                background: #f8f8f8;
                border-left: 4px solid #0066cc;
                padding: 16px 20px;
                margin: 16px 0;
                border-radius: 0 8px 8px 0;
            }
            .summary-box h3 {
                margin: 0 0 8px 0;
                color: #0066cc;
                font-size: 18px;
                font-weight: 500;
            }
            .summary-box p {
                margin: 0;
                color: #333;
                line-height: 1.6;
                font-size: 16px;
            }
            .model-info {
                font-style: italic;
                color: #666;
                font-size: 14px;
                margin-top: 12px;
                border-top: 1px solid #e5e5e5;
                padding-top: 12px;
            }
        '''
    
    def format_article_html(self, article, include_summary=True, for_email=False):
        """Format a single article as HTML."""
        template = '''
        <div class="article">
            <h2><a href="{link}" target="_blank">{title}</a></h2>
            <div class="{meta_class}">
                Published on {pub_date} by <a href="{link}" target="_blank">{source}</a>
            </div>
            {summary}
            {notes}
            {tags}
        </div>
        '''
        
        if for_email:
            article_class = 'article" style="margin-bottom: 30px; padding: 15px; border: 1px solid #eee;'
            meta_class = 'meta" style="font-size: 0.9em; color: #666; margin-bottom: 10px;'
            tag_style = 'display: inline-block; padding: 2px 8px; margin: 2px; background: #edf2f7; border-radius: 12px; font-size: 0.85em; color: #4a5568;'
        else:
            article_class = 'article'
            meta_class = 'meta'
            tag_style = None
            
        # Format tags
        if article.get('tags'):
            tags_html = '<div class="tags">'
            for tag in article['tags']:
                if for_email:
                    tags_html += f'<span style="{tag_style}">{tag}</span> '
                else:
                    tags_html += f'<span class="tag">{tag}</span> '
            tags_html += '</div>'
        else:
            tags_html = ''
        
        # Format notes if present
        notes_html = ''
        if article.get('notes'):
            if for_email:
                notes_html = f'<div style="margin-top: 10px; padding: 10px; background: #fffaf0; border-left: 3px solid #ed8936;">{article["notes"]}</div>'
            else:
                notes_html = f'<div class="notes">{article["notes"]}</div>'
        
        # Format summary
        summary_html = ''
        if include_summary and article.get('summary'):
            if isinstance(article['summary'], dict):
                headline = article['summary'].get('headline', '')
                summary = article['summary'].get('summary', '')
            else:
                headline = ''
                summary = article['summary']
            
            summary_html = f'''
            <div class="summary-box">
                <h3>{headline}</h3>
                <p>{summary}</p>
                <div class="model-info">
                    Model: {self.model} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </div>
            </div>
            '''
        
        # Format dates
        pub_date = ''
        if article.get('pub_date'):
            pub_date = article['pub_date']
        
        return template.format(
            article_class=article_class,
            meta_class=meta_class,
            link=article['link'],
            title=article['title'],
            pub_date=pub_date,
            source=article.get('source', 'Unknown Source'),
            summary=summary_html,
            notes=notes_html,
            tags=tags_html
        )
    
    def generate_html(self, articles, title, include_summary=True):
        """Generate full HTML document for articles."""
        content = ''.join(
            self.format_article_html(article, include_summary)
            for article in articles
        )
        return self.html_template.format(title=title, content=content)
    
    def generate_email_html(self, articles, title, include_summary=True):
        """Generate email-friendly HTML for articles."""
        content = ''.join(
            self.format_article_html(article, include_summary, for_email=True)
            for article in articles
        )
        return self.email_template.format(title=title, content=content)
    
    def generate_pdf(self, articles, title, output_file=None, include_summary=True):
        """Generate PDF document from articles.
        
        Args:
            articles (list): List of articles to include
            title (str): Document title
            output_file (str): Output file path
            include_summary (bool): Whether to include article summaries
            
        Returns:
            str: Path to generated PDF file
        """
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
        except ImportError:
            logging.error("WeasyPrint not installed. Installing required dependencies...")
            os.system('pip install weasyprint')
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
        
        # Generate HTML content
        html_content = self.generate_html(articles, title, include_summary)
        
        # Configure fonts
        font_config = FontConfiguration()
        
        # Create PDF
        if not output_file:
            output_file = f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        HTML(string=html_content).write_pdf(
            output_file,
            stylesheets=[
                CSS(string=self.pdf_css),
                CSS(string=self.summary_css),
                CSS(string='@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap");'),
            ],
            font_config=font_config
        )
        
        return output_file

class ReadwiseClient:
    """Client for interacting with Readwise Reader API."""
    
    def __init__(self, api_token=None):
        """Initialize Readwise client.
        
        Args:
            api_token (str): Readwise Reader API token
        """
        self.api_token = api_token or get_env_var('READWISE_TOKEN')
        if not self.api_token:
            raise ValueError("Readwise API token not provided")
        
        self.base_url = "https://readwise.io/api/v3"
        self.headers = {
            'Authorization': f'Token {self.api_token}',
            'Content-Type': 'application/json'
        }
    
    def save_article(self, article):
        """Save an article to Readwise Reader.
        
        Args:
            article (dict): Article to save
            
        Returns:
            dict: Response from Readwise API
        """
        url = f"{self.base_url}/save"
        
        # Prepare tags
        tags = []
        if article.get('tags'):
            tags = [tag.replace('company:', 'company/').replace('type:', 'type/')
                   for tag in article['tags']]
        
        # Prepare data
        data = {
            'url': article['link'],
            'title': article['title'],
            'summary': article.get('summary', ''),
            'notes': article.get('notes', ''),
            'tags': tags,
            'should_clean_html': True
        }
        
        try:
            response = requests.post(url, json=data, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error saving to Readwise: {e}")
            raise

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
            system_prompt = """You are an expert at creating concise, informative summaries of articles.
Your task is to summarize the following article:
<article>
{text}
</article>

Create a summary that adheres to these guidelines:

1. Length: Three to four sentences.

2. Style:
   - Use active voice
   - Choose non-compound verbs when possible
   - Avoid the words "content" and "creator"
   - Use "open source" instead of "open-source"
   - Spell out numbers (e.g., "8 billion" instead of "8B")
   - Abbreviate U.S. and U.K. with periods; use AI without periods
   - Use smart quotes, not straight quotes

3. Content structure:
   - First sentence: Explain what has happened in clear, simple language
   - Second sentence: Identify important details relevant to AI developers
   - Third sentence: Explain why this information matters to readers who closely follow AI news

4. Tone:
   - Factual, informative, and free from exaggeration, hype, or marketing speak

5. Headline:
   - Create a headline in sentence case
   - Avoid repeating too many words or phrases from the summary

After creating your summary, review it to ensure accuracy and remove any exaggerated language.
Start directly with the summary content - do not include phrases like "Here is a summary" or "In summary."

Provide your final output in the following format:
[Your headline here]
[Your summary here]"""

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            
            response = rate_limited_api_call(
                self.client,
                messages,
                model="claude-3-haiku-20240307",
                max_tokens=150
            )
            
            summary = response.content[0].text
            
            # Cache the summary
            self.summary_cache.set(text, {'summary': summary})
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return "Summary not available."

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
        
        # Initialize sentence transformer
        self.model = None
        self.device = None

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

    def _parse_entry(self, entry):
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
                'feed_source': getattr(entry, 'feed', {}).get('title', 'Unknown Source')
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
                
                # Generate summaries for articles without them
                for article in batch_articles:
                    if not article.get('summary'):
                        result = self._generate_summary(article['content'], article['title'], article['link'])
                        article['summary'] = result
                
                all_articles.extend(batch_articles)
                
                # Add delay between batches if there are more
                if batch['current'] < batch['total']:
                    time.sleep(self.batch_delay)
            
            # Cluster similar articles
            start_time = time.time()
            clusters = self._cluster_articles(all_articles) if all_articles else []
            duration = time.time() - start_time
            logging.info(f"Performance: cluster_similar_articles took {duration:.4f} seconds "
                        f"(CPU: {psutil.cpu_percent()}%, Memory: {psutil.Process().memory_info().rss} bytes)")
            
            # Generate HTML output
            if clusters:
                output_file = self.generate_html_output(clusters)
                return output_file
            
            return None
            
        except Exception as e:
            logging.error(f"Error processing feeds: {str(e)}")
            return None

    def _process_feed(self, feed_url):
        """Process a single RSS feed."""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            if feed.entries:
                logging.info(f"📰 Found {len(feed.entries)} articles in feed: {feed_url}")
                
                for entry in feed.entries[:self.batch_size]:
                    article = self._parse_entry(entry)
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
                "1. First line: Create a headline in sentence case\n"
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
            
            # Get embeddings for all articles
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Extract titles for clustering
            titles = [article.get('title', '') for article in articles]
            
            # Get embeddings for titles
            embeddings = model.encode(titles, show_progress_bar=True)
            
            # Cluster using DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=1).fit(embeddings)
            
            # Group articles by cluster
            clusters = {}
            for idx, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(articles[idx])
                
            # Convert to list of clusters
            clustered_articles = list(clusters.values())
            
            # Log performance metrics
            end_time = time.time()
            duration = end_time - start_time
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info().rss
            
            logging.info(f"Performance: cluster_similar_articles took {duration:.4f} seconds "
                        f"(CPU: {cpu_percent}%, Memory: {memory_info} bytes)")
            
            return clustered_articles
            
        except Exception as e:
            logging.error(f"Error clustering articles: {str(e)}")
            return []

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
        """
        Generate HTML output from the processed articles.
        
        Args:
            clusters (list): List of article clusters
            
        Returns:
            str: Path to the generated HTML file
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.getcwd(), 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'rss_summary_{timestamp}.html')
            
            # Generate content for each cluster
            content = ""
            for cluster in clusters:
                content += '<div class="cluster">'
                for article in cluster:
                    title = article.get('title', 'No Title')
                    link = article.get('link', '#')
                    summary = article.get('summary', {})
                    if isinstance(summary, dict):
                        headline = summary.get('headline', '')
                        summary_text = summary.get('summary', '')
                    else:
                        headline = ''
                        summary_text = str(summary)
                    published = article.get('published', 'Unknown date')
                    source = article.get('feed_source', 'Unknown source')
                    
                    content += f'''
        <div class="article">
            <h2 class="title"><a href="{link}" target="_blank">{title}</a></h2>
            <div class="metadata">
                Published: {published}<br>
                Source: {source}
            </div>
            <div class="summary">
                <h3>{headline}</h3>
                <p>{summary_text}</p>
            </div>
        </div>'''
                content += '\n    </div>'
            
            # HTML template
            html = f'''<!DOCTYPE html>
<html>
<head>
    <title>RSS Feed Summary</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .article {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .title {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .summary {{
            color: #34495e;
            margin: 15px 0;
        }}
        .metadata {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .cluster {{
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-bottom: 30px;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <h1>RSS Feed Summary</h1>
    <div class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    <div class="content">
{content}
    </div>
</body>
</html>'''
            
            # Write HTML file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logging.info(f"✅ Output written to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error generating HTML output: {str(e)}")
            return None

def test_readwise_token():
    """Test if Readwise token is properly loaded from .env"""
    try:
        client = ReadwiseClient()
        logging.info("Successfully loaded Readwise token from .env")
        # Test a simple API call to verify the token works
        response = requests.get(
            "https://readwise.io/api/v3/export",
            headers=client.headers
        )
        response.raise_for_status()
        logging.info("Successfully authenticated with Readwise API")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logging.error("Invalid Readwise token")
        else:
            logging.info("Successfully authenticated with Readwise API (token is valid)")
        return e.response.status_code != 401
    except Exception as e:
        logging.error(f"Error testing Readwise token: {e}")
        return False

def main():
    """
    Main function to run the RSS reader.
    """
    try:
        # Test Readwise token
        if not test_readwise_token():
            logging.error("❌ Failed to authenticate with Readwise API")
            return

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
