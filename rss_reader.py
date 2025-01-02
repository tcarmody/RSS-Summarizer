"""
RSS Reader with AI-Powered Summarization and Article Clustering
"""

import os
import re
import json
import time
import logging
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
from anthropic import Anthropic
from tqdm import tqdm
from dotenv import load_dotenv
from urllib.parse import urlparse

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
                import psutil
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
        self._load_cache()
    
    def _load_cache(self):
        """
        Load the cache from disk, creating an empty one if it doesn't exist.
        Also performs cache cleanup by removing expired entries.
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                # Clean up expired entries
                self._cleanup_cache()
            else:
                self.cache = {}
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """
        Save the current cache to disk in JSON format.
        Implements basic error handling to prevent data loss.
        """
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")
    
    def _cleanup_cache(self):
        """
        Remove expired entries and enforce maximum cache size.
        Uses LRU (Least Recently Used) strategy when cache exceeds max size.
        """
        current_time = time.time()
        # Remove expired entries
        self.cache = {
            k: v for k, v in self.cache.items()
            if current_time - v.get('timestamp', 0) < self.cache_duration
        }
        
        # If still too large, remove oldest entries
        if len(self.cache) > self.max_cache_size:
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            self.cache = dict(sorted_items[-self.max_cache_size:])
    
    def get(self, key: str) -> Optional[Dict]:
        """
        Retrieve a summary from the cache.
        
        Args:
            key (str): Cache key (usually URL or content hash)
        
        Returns:
            Optional[Dict]: Cached summary if found and not expired, None otherwise
        """
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry.get('timestamp', 0) < self.cache_duration:
                return entry
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Dict):
        """
        Store a summary in the cache.
        
        Args:
            key (str): Cache key (usually URL or content hash)
            value (Dict): Summary data to cache
        """
        value['timestamp'] = time.time()
        self.cache[key] = value
        if len(self.cache) > self.max_cache_size:
            self._cleanup_cache()
        self._save_cache()
    
    def _hash_text(self, text):
        """
        Generate a hash for the given text to use as a cache key.
        
        Args:
            text (str or set): Text to hash
        
        Returns:
            str: Hashed text
        """
        import hashlib
        # Convert text to string if it's not already
        if isinstance(text, (set, list, tuple)):
            # Convert any iterable to a sorted list of strings
            text = ' '.join(sorted(str(item) for item in text))
        elif not isinstance(text, str):
            text = str(text)
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text, force_refresh=False):
        """
        Retrieve cached summary for a given text.
        
        :param text: Original text to summarize
        :param force_refresh: If True, ignore cached version
        :return: Cached summary or None if not found/expired
        """
        if force_refresh:
            return None
        
        # Convert text to string if it's not already
        if isinstance(text, (set, list, tuple)):
            text = ' '.join(sorted(str(item) for item in text))
        elif not isinstance(text, str):
            text = str(text)
        
        text_hash = self._hash_text(text)
        current_time = datetime.now(pytz.UTC)
        
        # Check if entry exists and is not expired
        if text_hash in self.cache:
            entry = self.cache[text_hash]
            if 'timestamp' in entry:
                entry_time = datetime.fromtimestamp(entry['timestamp'], pytz.UTC)
                if current_time - entry_time < timedelta(seconds=self.cache_duration):
                    # Convert any sets in the summary back to lists
                    if isinstance(entry, dict) and 'summary' in entry:
                        for field, value in entry['summary'].items():
                            if isinstance(value, set):
                                entry['summary'][field] = list(value)
                    return entry
        
        return None
    
    def set(self, text, summary, force_update=False):
        """
        Cache a summary for a given text.
        
        :param text: Original text
        :param summary: Summary dictionary
        :param force_update: If True, update even if entry exists
        """
        # Convert text to string if it's not already
        if isinstance(text, (set, list, tuple)):
            text = ' '.join(sorted(str(item) for item in text))
        elif not isinstance(text, str):
            text = str(text)
        
        text_hash = self._hash_text(text)
        if text_hash not in self.cache or force_update:
            self.cache[text_hash] = {
                'timestamp': time.time(),
                'summary': summary
            }
            self._save_cache()
    
    def clear_cache(self):
        """
        Completely clear the cache.
        """
        self.cache.clear()
        try:
            os.remove(self.cache_file)
        except FileNotFoundError:
            pass
        
        # Recreate cache file
        with open(self.cache_file, 'w') as f:
            json.dump({}, f)
        
        print("Summary cache has been cleared.")

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
        
        # Compile all patterns for better performance
        self.sponsored_regex = re.compile('|'.join(self.sponsored_patterns))
        self.crypto_regex = re.compile('|'.join(self.crypto_patterns))
        
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
                    max-width: 800px;
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
    
    def format_article_html(self, article, include_summary=True, for_email=False):
        """Format a single article as HTML."""
        template = '''
        <div class="{article_class}">
            <h2><a href="{link}" target="_blank">{title}</a></h2>
            <div class="{meta_class}">
                Published: {pub_date}
                {added_date}
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
            summary_html = f'<p>{article["summary"]}</p>'
        
        # Format dates
        added_date = ''
        if article.get('added_date'):
            added_date = f' | Added: {article["added_date"]}'
        
        return template.format(
            article_class=article_class,
            meta_class=meta_class,
            link=article['link'],
            title=article['title'],
            pub_date=article.get('pub_date', 'Unknown'),
            added_date=added_date,
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
    
    def __init__(self, api_key=None, cache_dir='.cache'):
        """
        Initialize the RSS reader with API credentials and caching.
        
        Args:
            api_key (str): Anthropic API key for Claude
            cache_dir (str): Directory for caching summaries and feed data
        """
        self.anthropic = Anthropic(api_key=api_key or get_env_var('ANTHROPIC_API_KEY'))
        self.model = "claude-3-haiku-20240307"
        self.cache = SummaryCache(cache_dir=cache_dir)
        self._sentence_transformer = None  # Lazy-loaded for performance
        self.favorites = FavoritesManager(cache_dir)  # Initialize favorites manager
        
        # Configure batch processing
        self.batch_size = 25
        self.batch_delay = 15  # seconds between batches
        
        # List of domains that are likely to be paywalled
        self.protected_domains = [
            'wsj.com',
            'nytimes.com',
            'bloomberg.com',
            'ft.com',
            'economist.com'
        ]
        
        # Configure user agent and headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Initialize feed processing settings
        self.config = RSSReaderConfig()
        
        # Initialize feed cache
        self.feed_cache = FeedCache()
    
    def _get_sentence_transformer(self):
        """
        Lazy-load the sentence transformer model for semantic similarity.
        Uses a lightweight but effective model for clustering.
        """
        if self._sentence_transformer is None:
            logging.info("Load pretrained SentenceTransformer: all-MiniLM-L6-v2")
            self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        return self._sentence_transformer
    
    @track_performance()
    def process_feeds(self, feed_urls, batch_size=25, batch_delay=15):
        """
        Process a list of RSS feeds in batches.
        
        Args:
            feed_urls (List[str]): List of RSS feed URLs to process
            batch_size (int): Number of articles to process in each batch
            batch_delay (int): Delay between batches in seconds
        
        Returns:
            List[Dict]: List of processed articles with summaries
        """
        logging.info(f"📊 Total Feeds: {len(feed_urls)}")
        logging.info(f"📦 Batch Size: {batch_size}")
        logging.info(f"⏱️  Batch Delay: {batch_delay} seconds")
        
        total_batches = (len(feed_urls) + batch_size - 1) // batch_size
        logging.info(f"🔄 Total Batches: {total_batches}")
        
        all_articles = []
        for i in range(0, len(feed_urls), batch_size):
            batch = feed_urls[i:i + batch_size]
            logging.info(f"\n🔄 Processing Batch {i//batch_size + 1}/{total_batches}: Feeds {i+1} to {min(i+batch_size, len(feed_urls))}")
            
            for url in batch:
                try:
                    # Parse the feed
                    parsed_feed = feedparser.parse(url)
                    if parsed_feed.bozo:
                        logging.error(f"❌ Error parsing feed {url}: {parsed_feed.bozo_exception}")
                        continue
                    
                    logging.info(f"📰 Found {len(parsed_feed.entries)} articles in feed: {url}")
                    
                    # Process each entry in the feed
                    for entry in parsed_feed.entries:
                        try:
                            # Get feed title from multiple possible locations
                            feed_title = (
                                parsed_feed.feed.get('title') or
                                parsed_feed.feed.get('subtitle') or
                                parsed_feed.feed.get('description') or
                                parsed_feed.feed.get('link') or
                                self.get_publication_name(url)
                            )
                            
                            # Clean up feed title if it's a URL
                            if feed_title and (feed_title.startswith('http://') or feed_title.startswith('https://')):
                                feed_title = self.get_publication_name(feed_title)
                            
                            # Add feed source to the entry
                            entry['feed_source'] = feed_title
                            all_articles.append(entry)
                        except Exception as e:
                            logging.error(f"❌ Error processing entry from {url}: {e}")
                    
                except Exception as e:
                    logging.error(f"❌ Error processing feed {url}: {e}")
            
            if i + batch_size < len(feed_urls):
                time.sleep(batch_delay)
        
        return all_articles
    
    def get_cache_key(self, url):
        """Generate a cache key for a URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def get_from_cache(self, url):
        """Retrieve content from cache if available and not expired."""
        cache_key = self.get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Parse timestamp with timezone info
                cached_time = datetime.fromisoformat(cached_data['timestamp'])
                if isinstance(cached_time, datetime) and cached_time.tzinfo is None:
                    cached_time = self.eastern_tz.localize(cached_time)
                
                # Check if cache is still valid (24 hours)
                current_time = self.get_current_eastern_time()
                if current_time - cached_time < timedelta(hours=24):
                    print(f"Retrieved from cache: {url}")
                    return cached_data['content']
        except Exception as e:
            print(f"Error reading cache: {e}")
        
        return None

    def save_to_cache(self, url, content):
        """Save content to cache."""
        if not content:
            return
            
        cache_key = self.get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            cache_data = {
                'url': url,
                'content': content,
                'timestamp': self.get_current_eastern_time().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving to cache: {e}")

    @RateLimiter(max_calls_per_minute=20)
    def make_request_with_retry(self, url, max_retries=3, initial_delay=1):
        """Make HTTP request with exponential backoff retry."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                delay = delay  # Add jitter
                # time.sleep(delay + jitter)
        
        return None

    def try_archive_service(self, service_url, url):
        """Try to fetch content from a specific archive service."""
        try:
            archive_url = service_url + url
            response = self.make_request_with_retry(archive_url)
            
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try to find main content
                for selector in ['article', 'main', '.article-body', '.story-body', '.post-content']:
                    content = soup.select_one(selector)
                    text = self.extract_text_content(content)
                    if text:
                        print(f"Successfully retrieved content from {service_url}")
                        return text
            
            return None
        except Exception as e:
            print(f"Error checking {service_url}: {e}")
            return None

    def get_article_content(self, entry, feed_entry=None):
        """Get the full article content from the URL."""
        try:
            url = entry.get('link', '')
            if not url:
                return None, None

            # Check if URL is in cache
            cached_content = self.get_from_cache(url)
            if cached_content:
                # Create BeautifulSoup object for cached content
                soup = BeautifulSoup(cached_content, 'html.parser')
                return cached_content, soup

            # Check if URL is from a protected domain
            if self.is_protected_content(url):
                print(f"🔒 Attempting to bypass paywall for {url}")
                # Try archive services
                for service in self.archive_services:
                    archive_content = self.try_archive_service(service, url)
                    if archive_content:
                        soup = BeautifulSoup(archive_content, 'html.parser')
                        # Save to cache
                        self.save_to_cache(url, archive_content)
                        return archive_content, soup

            # Fetch article content
            content, soup = self.fetch_article_content(url)

            # Save to cache if content is retrieved
            if content:
                self.save_to_cache(url, content)

            return content, soup

        except Exception as e:
            print(f"Error getting article content: {e}")
            return None, None

    def fetch_article_content(self, url):
        """
        Fetch the main content of an article from a given URL, trying various bypass methods.
        
        Args:
            url (str): URL of the article to fetch
        
        Returns:
            tuple: (article_text, BeautifulSoup object)
        """
        try:
            # Skip paywalled sites
            if any(domain in url.lower() for domain in self.protected_domains):
                logger.warning(f"Protected content detected, trying bypass methods: {url}")
                return None, None

            # Try direct fetch first
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                content, soup = self.extract_article_content(response.text)
                if content:
                    return content, soup
                    
            # If direct fetch fails or no content found, try bypass methods
            return self.fetch_with_bypass(url)
            
        except requests.RequestException as e:
            logger.error(f"Request error fetching article from {url}: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Error fetching article from {url}: {str(e)}")
            return None, None

    def process_feed_entry(self, entry):
        """Process a single feed entry."""
        try:
            # Extract article URL
            url = entry.get('link', '')
            if not url:
                logger.warning("Entry has no URL")
                return None
                
            # Get the article content
            content = entry.get('content', [{}])[0].get('value', '') or entry.get('summary', '')
            if not content:
                # If no content in feed, try fetching from URL
                content, _ = self.fetch_article_content(url)
                if not content:
                    logger.warning(f"No content found for article: {url}")
                    return None
            
            # Clean and summarize content if needed
            summary = None
            if len(content) > 50:  # Lower threshold to ensure we get summaries
                try:
                    summary = self.summarize_text(content, entry.get('title'))
                    logging.info(f"Generated summary for article: {entry.get('title', 'Unknown title')}")
                except Exception as e:
                    logging.error(f"Failed to generate summary: {str(e)}")
            
            # Get publication date
            published = entry.get('published', entry.get('updated', 'No date available'))
            
            # Extract source name
            source = entry.get('feed_source')
            if not source:
                source = self.get_publication_name(url, feed_entry=entry)
                if source == "Unknown source":
                    # Try harder to get a source name
                    source = self.get_publication_name(url)
        
            return {
                'title': entry.get('title', 'No title available'),
                'link': url,
                'published': published,
                'content': content,
                'summary': summary if summary else None,  # Store the entire summary dictionary
                'model': summary.get('model', 'Fallback') if summary else 'Fallback',
                'source': source  # Use the improved source extraction
            }
            
        except Exception as e:
            logger.error(f"Error processing feed entry {entry.get('link', 'Unknown URL')}: {str(e)}")
            return None

    def write_to_html(self, articles, output_file=None):
        """Write articles to an HTML file."""
        try:
            if not articles:
                logger.warning("No articles to write to HTML")
                return None
                
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(output_dir, f'rss_summary_{timestamp}.html')
            
            # Generate HTML content
            html_content = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>AI News Summary</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                        line-height: 1.6;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }
                    h1 {
                        color: #333;
                        border-bottom: 2px solid #ddd;
                        padding-bottom: 10px;
                    }
                    .article {
                        background: white;
                        border-radius: 8px;
                        padding: 25px;
                        margin-bottom: 30px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .article.cluster {
                        border-left: 4px solid #3498db;
                    }
                    .article h2 {
                        color: #2c3e50;
                        margin-top: 0;
                        margin-bottom: 15px;
                    }
                    .article-meta {
                        color: #7f8c8d;
                        font-size: 0.9em;
                        margin-bottom: 20px;
                    }
                    .article-summary {
                        background: #f8f9fa;
                        border-radius: 6px;
                        padding: 20px;
                        margin-bottom: 20px;
                    }
                    .summary-header {
                        margin-bottom: 15px;
                        color: #2c3e50;
                    }
                    .summary-content {
                        margin-bottom: 15px;
                        line-height: 1.7;
                    }
                    .summary-footer {
                        color: #7f8c8d;
                        font-size: 0.9em;
                        display: flex;
                        justify-content: space-between;
                        border-top: 1px solid #eee;
                        padding-top: 15px;
                        margin-top: 15px;
                    }
                    .article-footer {
                        margin-top: 20px;
                    }
                    .article-link {
                        color: #3498db;
                        text-decoration: none;
                    }
                    .article-link:hover {
                        color: #2980b9;
                        text-decoration: underline;
                    }
                    .article-sources {
                        margin-top: 20px;
                        padding-top: 20px;
                        border-top: 1px solid #eee;
                    }
                    .article-sources h3 {
                        color: #2c3e50;
                        margin-top: 0;
                        margin-bottom: 15px;
                        font-size: 1.1em;
                    }
                    .source-list {
                        list-style: none;
                        padding: 0;
                        margin: 0;
                    }
                    .source-list li {
                        margin-bottom: 10px;
                        padding-left: 20px;
                        position: relative;
                    }
                    .source-list li:before {
                        content: "•";
                        color: #3498db;
                        position: absolute;
                        left: 0;
                    }
                    .model-info, .timestamp-info {
                        display: inline-block;
                        padding: 4px 8px;
                        border-radius: 4px;
                        background: #e8e8e8;
                        font-size: 0.85em;
                    }
                </style>
            </head>
            <body>
                <h1>AI News Summary</h1>
            '''
            
            # Process and write clusters
            clusters = self.cluster_similar_articles(articles)
            for cluster in clusters:
                if len(cluster) == 1:
                    # Single article - display as before
                    article = cluster[0]
                    summary_dict = article.get('summary', {})
                    
                    if isinstance(summary_dict, str):
                        try:
                            summary_dict = json.loads(summary_dict)
                        except:
                            summary_dict = {'headline': 'No Headline', 'summary': summary_dict}
                    
                    headline = summary_dict.get('headline', article.get('title', 'No headline available'))
                    summary_text = summary_dict.get('summary', 'No summary available')
                    model = summary_dict.get('model', 'Unknown')
                    timestamp = summary_dict.get('timestamp')
                    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'Unknown'
                    
                    html_content += f'''
                    <div class="article">
                        <h2>{html.escape(headline)}</h2>
                        <div class="article-meta">
                            From <a href="{html.escape(str(article.get('link', '#')))}" target="_blank">{html.escape(str(article.get('feed_source', 'Unknown Source')))}</a>
                        </div>
                        <div class="article-summary">{html.escape(summary_text)}</div>
                        <div class="metadata">
                            Model: {html.escape(model)} | Generated: {html.escape(timestamp_str)}
                        </div>
                    </div>
                    '''
                else:
                    # Multiple similar articles - generate combined summary
                    summary = self.generate_cluster_summary(cluster)
                    if summary:
                        if isinstance(summary, str):
                            try:
                                summary = json.loads(summary)
                            except:
                                summary = {'headline': 'No Headline', 'summary': summary}
                        
                        headline = summary.get('headline', 'Related Articles')
                        summary_text = summary.get('summary', 'No summary available')
                        model = summary.get('model', 'Unknown')
                        timestamp = summary.get('timestamp')
                        timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'Unknown'
                        
                        # Get sources from the cluster
                        source_list = []
                        for article in cluster:
                            source_list.append({
                                'title': article.get('title', 'No title'),
                                'link': article['link'],
                                'source': article.get('feed_source', 'Unknown Source')
                            })
                        
                        html_content += f'''
                        <div class="article cluster">
                            <h2>{html.escape(headline)}</h2>
                            <div class="article-meta">
                                From multiple sources:
                                <ul class="source-list">
                        '''
                        
                        for source in source_list:
                            html_content += f'''
                                <li>
                                    <a href="{html.escape(source['link'])}" target="_blank">
                                        {html.escape(source['title'])} ({html.escape(source['source'])})
                                    </a>
                                </li>
                            '''
                        
                        html_content += f'''
                                </ul>
                            </div>
                            <div class="article-summary">{html.escape(summary_text)}</div>
                            <div class="metadata">
                                Model: {html.escape(model)} | Generated: {html.escape(str(timestamp_str))}
                            </div>
                        </div>
                        '''
            
            # Add footer
            html_content += f'''
                <div class="footer">
                    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} using Claude</p>
                </div>
            </body>
            </html>
            '''
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"✅ HTML output written to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating HTML for article: {str(e)}")
            return None
            
    def generate_concise_summary(self, text):
        """Generate a concise 3-4 sentence summary following the style guide."""
        try:
            if not text:
                return None
                
            prompt = """You are an expert at creating summaries of articles. Summaries will be three to four sentences in length. Summaries should be factual, informative, fairly simple in structure, and free from exaggeration, hype, or marketing speak.

STYLE: 
- Avoid passive voice
- Choose non-compound verbs whenever possible
- Avoid the words "content" and "creator"
- Instead of "open-source," use "open source"
- Spell out numbers (e.g., "8 billion" not "8B")
- Use "U.S." and "U.K." with periods; use "AI" without periods
- Use smart quotes, not straight quotes

The first sentence should explain what has happened in clear, simple language.
The second sentence should identify important details that are relevant to an audience of AI developers.
The third sentence should explain why this information matters to an audience of readers who closely follow AI news.

Please summarize the following text in this style:

{text}
"""
            
            # Call Claude API directly for summary generation
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            if response and response.content:
                # Extract just the summary text, removing any "Here's a summary:" prefix
                summary_text = response.content[0].text
                if "Here's a summary:" in summary_text:
                    summary_text = summary_text.split("Here's a summary:", 1)[1].strip()
                elif "Here's a concise summary:" in summary_text:
                    summary_text = summary_text.split("Here's a concise summary:", 1)[1].strip()
                    
                return summary_text
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating concise summary: {str(e)}")
            return None
    
    def fetch_articles(self, feeds_file='rss_test.txt'):
        """
        Fetch articles from multiple RSS feeds.
        
        Args:
            feeds_file (str): Path to file containing RSS feed URLs
        
        Returns:
            list: Collected articles from all feeds
        """
        # Read all feeds from the file
        with open(feeds_file, 'r') as f:
            feed_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Collect articles from all feeds
        all_articles = []
        
        # Process each feed
        for url in feed_urls:
            try:
                # Sanitize the feed URL
                url = url.split('#')[0].strip()  # Remove comments and strip whitespace
                url = ''.join(char for char in url if ord(char) >= 32)  # Remove control characters
                
                # Parse the feed
                parsed_feed = feedparser.parse(url)
                
                # Validate feed
                if not parsed_feed.entries:
                    print(f"⚠️ No entries found in feed: {url}")
                    continue
                
                # Process each entry in the feed
                for entry in parsed_feed.entries:
                    try:
                        # Get feed title from multiple possible locations
                        feed_title = (
                            parsed_feed.feed.get('title') or
                            parsed_feed.feed.get('subtitle') or
                            parsed_feed.feed.get('description') or
                            parsed_feed.feed.get('link') or
                            self.get_publication_name(url)
                        )
                        
                        # Clean up feed title if it's a URL
                        if feed_title and (feed_title.startswith('http://') or feed_title.startswith('https://')):
                            feed_title = self.get_publication_name(feed_title)
                        
                        # Add feed source to the entry
                        entry['feed_source'] = feed_title
                        all_articles.append(entry)
                    except Exception as e:
                        print(f"❌ Error processing entry from {url}: {e}")
            
            except Exception as e:
                print(f"❌ Error parsing feed {url}: {e}")
        
        # Sort articles by publication date in descending order
        all_articles.sort(
            key=lambda x: x.get('published_parsed', datetime.min), 
            reverse=True
        )
        
        return all_articles

    @track_performance()
    def process_feeds(self, feeds_file='rss_test.txt'):
        """Process RSS feeds and generate summaries."""
        try:
            # Load feed URLs
            feed_urls = self.load_feed_urls(feeds_file)
            if not feed_urls:
                logger.error("No feed URLs found")
                return None
                
            logger.info(f"📊 Total Feeds: {len(feed_urls)}")
            logger.info(f"📦 Batch Size: {self.batch_size}")
            logger.info(f"⏱️  Batch Delay: {self.batch_delay} seconds")
            
            total_batches = (len(feed_urls) + self.batch_size - 1) // self.batch_size
            logger.info(f"🔄 Total Batches: {total_batches}")
            
            all_articles = []
            batch_start_time = time.time()
            
            for batch_num in range(total_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(feed_urls))
                batch_urls = feed_urls[start_idx:end_idx]
                
                logger.info(f"\n🔄 Processing Batch {batch_num + 1}/{total_batches}: Feeds {start_idx + 1} to {end_idx}")
                
                for feed_url in batch_urls:
                    try:
                        # Parse the RSS feed
                        feed = feedparser.parse(feed_url)
                        
                        if not feed.entries:
                            logger.warning(f"No entries found in feed: {feed_url}")
                            continue
                            
                        logger.info(f"📰 Found {len(feed.entries)} articles in feed: {feed_url}")
                        
                        # Process each article in the feed
                        for entry in feed.entries:
                            try:
                                article = self.process_feed_entry(entry)
                                if article:
                                    all_articles.append(article)
                            except Exception as e:
                                logger.error(f"❌ Failed to process entry: {entry.get('link', 'Unknown URL')}\n{str(e)}")
                                continue
                                
                    except Exception as e:
                        logger.error(f"❌ Failed to process feed: {feed_url}\n{str(e)}")
                        continue
                
                batch_time = time.time() - batch_start_time
                logger.info(f"⏱️  Batch {batch_num + 1} Processing Time: {batch_time:.2f} seconds")
                
                # Add delay between batches if not the last batch
                if batch_num < total_batches - 1:
                    time.sleep(self.batch_delay)
            
            # Log final statistics
            total_time = time.time() - batch_start_time
            logger.info("\n🎉 Processing Complete!")
            logger.info(f"📊 Total Articles Processed: {len(all_articles)}")
            logger.info(f"❌ Failed Articles: {sum(len(feed.entries) for feed in (feedparser.parse(url) for url in feed_urls)) - len(all_articles)}")
            logger.info(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
            if all_articles:
                logger.info(f"📈 Average Processing Time per Article: {total_time/len(all_articles):.2f} seconds")
            
            return all_articles
            
        except Exception as e:
            logger.error(f"Error in process_feeds: {str(e)}")
            raise

    def load_feed_urls(self, feeds_file):
        """Load feed URLs from a file."""
        try:
            if not os.path.exists(feeds_file):
                logger.error(f"Feed file not found: {feeds_file}")
                return []
                
            with open(feeds_file, 'r') as f:
                # Read lines and clean them
                urls = []
                for line in f:
                    # Remove comments and whitespace
                    url = line.split('#')[0].strip()
                    if url:  # Skip empty lines
                        urls.append(url)
                        
                if not urls:
                    logger.warning("No URLs found in feed file")
                    
                return urls
        except Exception as e:
            logger.error(f"Error loading feed URLs: {str(e)}")
            return []

    def process_article(self, url):
        """Process a single article."""
        try:
            # Extract article content
            article_text, soup = self.fetch_article_content(url)
            
            if not article_text:
                logging.error(f"Could not extract content from {url}")
                return None
                
            # Get article title
            title = soup.title.string if soup and soup.title else "No title available"
            
            # Generate summary
            try:
                summary_dict = self.summarize_text(article_text, title)
                if not summary_dict:
                    summary_dict = {
                        'headline': title,
                        'summary': "Failed to generate summary",
                        'model': 'Error',
                        'timestamp': str(int(time.time()))
                    }
                logging.info(f"Generated summary for article: {title}")
            except Exception as e:
                logging.error(f"Error generating summary for {url}: {str(e)}")
                summary_dict = {
                    'headline': title,
                    'summary': "Summary generation failed",
                    'model': 'Error',
                    'timestamp': str(int(time.time()))
                }
                
            return {
                'title': title,
                'url': url,
                'content': article_text,
                'summary': summary_dict,
                'published': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error processing article {url}: {str(e)}")
            return None

    def fetch_article_content(self, url):
        """
        Fetch the main content of an article from a given URL.
        
        Args:
            url (str): URL of the article to fetch
        
        Returns:
            tuple: (article_text, BeautifulSoup object)
        """
        try:
            # Skip paywalled sites
            if any(domain in url.lower() for domain in self.protected_domains):
                logger.warning(f"Content may be protected/paywalled: {url}")
                return None, None

            # Fetch the webpage
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try various methods to extract article text
            article_text = ""
            
            # Try extracting text from specific article-related tags
            article_selectors = [
                'article', 
                'main', 
                '.article-body', 
                '.story-body', 
                '.post-content', 
                '#article-body', 
                '.content', 
                '.entry-content'
            ]
            
            for selector in article_selectors:
                content = soup.select_one(selector)
                if content:
                    # Extract text from paragraphs within the selected content
                    paragraphs = content.find_all('p')
                    article_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    
                    # If we found some text, break the loop
                    if article_text:
                        break
            
            # If no text found, fall back to all paragraphs
            if not article_text:
                paragraphs = soup.find_all('p')
                article_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            if not article_text:
                logger.warning(f"No article text found for {url}")
                return None, None
                
            # Clean up the text
            article_text = re.sub(r'\s+', ' ', article_text)  # Replace multiple spaces
            article_text = re.sub(r'\n\s*\n', '\n\n', article_text)  # Fix multiple newlines
            article_text = article_text.strip()
            
            # Truncate to a reasonable length
            article_text = article_text[:10000]
            
            return article_text, soup
        
        except requests.RequestException as e:
            logger.error(f"Request error fetching article from {url}: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Error fetching article from {url}: {str(e)}")
            return None, None

    def get_current_eastern_time(self):
        """Get the current time in Eastern timezone."""
        return datetime.now()

    def parse_date(self, date_str):
        """
        Parse date string from RSS feed with multiple fallback methods.

        Args:
            date_str (str): Date string to parse

        Returns:
            datetime: Parsed datetime object or current time if parsing fails
        """
        try:
            # Try parsing with dateutil first (most flexible)
            from dateutil.parser import parse

            # First, try parsing directly
            try:
                parsed_date = parse(date_str)

                # If no timezone, localize to Eastern
                if parsed_date.tzinfo is None:
                    parsed_date = self.eastern_tz.localize(parsed_date)

                return parsed_date
            except Exception:
                pass

            # Try parsing from email utils (for RFC 2822 format)
            try:
                parsed_date = parsedate_to_datetime(date_str)

                # If no timezone, localize to Eastern
                if parsed_date.tzinfo is None:
                    parsed_date = self.eastern_tz.localize(parsed_date)

                return parsed_date
            except Exception:
                pass

            # Fallback to current time
            return self.get_current_eastern_time()

        except ImportError:
            # If dateutil is not available, use more basic parsing
            try:
                # Try parsing ISO format
                try:
                    parsed_date = datetime.fromisoformat(date_str)

                    # If no timezone, localize to Eastern
                    if parsed_date.tzinfo is None:
                        parsed_date = self.eastern_tz.localize(parsed_date)

                    return parsed_date
                except Exception:
                    pass

                # Fallback to current time
                return self.get_current_eastern_time()

            except Exception:
                # Absolute last resort
                return self.get_current_eastern_time()

    def is_within_last_seven_days(self, pub_date):
        if pub_date is None:
            return False
        
        # Convert both datetimes to UTC if they have different timezones
        current_time = datetime.now(timezone.utc)
        pub_date = pub_date.astimezone(timezone.utc) if pub_date.tzinfo else current_time.replace(tzinfo=None)
        
        # Compare only the datetime values
        time_diff = current_time - pub_date
        return time_diff.total_seconds() <= 7 * 24 * 60 * 60  # 7 days

    def get_topic_keywords(self, text):
        """Extract main keywords from text."""
        # Ensure text is a string
        if not text:
            return []

        # Simple word frequency counting
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Return top 10 keywords as a list
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]  # Return just the words, not the counts

    def calculate_similarity(self, keywords1, keywords2):
        """Calculate similarity between two lists of keywords using Jaccard index."""
        # Convert to lists if not already
        if not isinstance(keywords1, list):
            keywords1 = list(keywords1)
        if not isinstance(keywords2, list):
            keywords2 = list(keywords2)
            
        # Count common elements
        common = len([x for x in keywords1 if x in keywords2])
        total = len(keywords1) + len(keywords2) - common
        
        # Avoid division by zero
        if total == 0:
            return 0
            
        return common / total

    @track_performance()
    def cluster_similar_articles(self, articles, similarity_threshold=0.5, max_clusters=10):
        """
        Advanced clustering of articles based on multi-dimensional semantic similarity.
        
        Uses sentence transformers to compute embeddings and cosine similarity
        to group related articles together. This helps reduce redundancy and
        provides a more comprehensive view of each news story.
        
        Args:
            articles (list): List of article dictionaries
            similarity_threshold (float): Threshold for clustering similar articles
            max_clusters (int): Maximum number of clusters to generate
        
        Returns:
            list: List of article clusters, where each cluster is a list of related articles
        """
        if not articles:
            logging.warning("No articles provided for clustering")
            return []
            
        try:
            # Extract titles and summaries for comparison
            texts = []
            for article in articles:
                title = article.get('title', '')
                summary_dict = article.get('summary', {})
                if not isinstance(summary_dict, dict):
                    summary_dict = {}
                summary = summary_dict.get('summary', '') if summary_dict else ''
                combined_text = f"{title} {summary}".strip()
                texts.append(combined_text)
                
            # Get embeddings using sentence transformer
            embeddings = self._get_sentence_transformer().encode(texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Initialize clusters
            clusters = []
            used_indices = set()  # Use a set for better performance
            
            # Create clusters based on similarity
            for i in range(len(articles)):
                if i in used_indices:
                    continue
                    
                # Start new cluster
                cluster = [articles[i]]
                used_indices.add(i)
                
                # Find similar articles
                for j in range(i + 1, len(articles)):
                    if j in used_indices:
                        continue
                        
                    if similarity_matrix[i][j] >= similarity_threshold:
                        cluster.append(articles[j])
                        used_indices.add(j)
                        
                clusters.append(cluster)
                
                # Stop if we've reached max clusters
                if len(clusters) >= max_clusters:
                    break
                    
            # Add remaining articles as single-article clusters
            for i in range(len(articles)):
                if i not in used_indices:
                    clusters.append([articles[i]])
                    
            return clusters
            
        except Exception as e:
            logging.error(f"Error clustering articles: {str(e)}")
            # Return each article as its own cluster
            return [[article] for article in articles]

    @track_performance()
    def generate_html_output(self, clusters, output_dir='output'):
        """Generate HTML output for the clusters."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'rss_summary_{timestamp}.html')
            
            # Start HTML content
            html_content = '''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>AI News Summary</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                /* ... rest of CSS styles ... */
            </style>
        </head>
        <body>
            <h1>AI News Summary</h1>
            '''
            
            # Process each cluster
            for cluster in clusters:
                try:
                    html_content += self._process_cluster(cluster)
                except Exception as cluster_error:
                    logger.error(f"Error processing cluster: {str(cluster_error)}")
                    continue
        
            # Add footer and close HTML
            html_content += self._generate_html_footer()
        
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"✅ Output written to {output_file}")
            return output_file
        
        except Exception as e:
            logger.error(f"Error generating HTML output: {str(e)}")
            if html_content:
                # Try to save what we have so far
                try:
                    error_file = os.path.join(output_dir, f'rss_summary_error_{timestamp}.html')
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    logger.info(f"Partial output saved to {error_file}")
                except Exception as save_error:
                    logger.error(f"Could not save partial output: {str(save_error)}")
            return None

    def exponential_backoff(self, attempt):
        """Generate exponential backoff with jitter."""
        import random
        base_delay = min(60, (2 ** attempt) + random.random())
        return base_delay
    
    def get_publication_name(self, url, feed_entry=None, article_soup=None):
        """Extract publication name from URL or feed entry."""
        # First try to get from feed entry
        if feed_entry and feed_entry.get('feed_source'):
            return feed_entry['feed_source']
        
        if article_soup:
            # Try to find publication name in meta tags
            meta_publishers = [
                article_soup.find('meta', attrs={'name': 'publisher'}),
                article_soup.find('meta', attrs={'property': 'og:site_name'}),
                article_soup.find('meta', attrs={'name': 'application-name'})
            ]
            for meta in meta_publishers:
                if meta and meta.get('content'):
                    return meta.get('content')
        
        # Fallback to URL-based extraction
        try:
            domain = urlparse(url).netloc
            # Remove www. if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Split by dots and get the main part
            parts = domain.split('.')
            if len(parts) >= 2:
                # Handle common domains directly
                known_domains = {
                    'engadget': 'Engadget',
                    'techcrunch': 'TechCrunch',
                    'forbes': 'Forbes',
                    'gizbot': 'GizBot',
                    'digit': 'Digit',
                    'timesnownews': 'Times Now News',
                    'gadgets360': 'Gadgets 360',
                    'verdict': 'Verdict'
                }
                
                if parts[0] in known_domains:
                    return known_domains[parts[0]]
                
                # Handle cases like 'theverge.com', 'techcrunch.com'
                if parts[0] in ['the', 'a', 'an']:
                    return f"{parts[0].capitalize()}{parts[1].capitalize()}"
                # Handle cases like 'blogs.nvidia.com'
                elif len(parts) > 2 and parts[0] in ['blog', 'blogs', 'news']:
                    return parts[1].capitalize()
                else:
                    # Try to make a readable name from the domain
                    name = parts[0]
                    # Split by common separators
                    if '-' in name:
                        words = name.split('-')
                        return ' '.join(word.capitalize() for word in words)
                    # Handle camelCase
                    elif any(c.isupper() for c in name[1:]):
                        words = []
                        current_word = name[0]
                        for char in name[1:]:
                            if char.isupper():
                                words.append(current_word)
                                current_word = char
                            else:
                                current_word += char
                        words.append(current_word)
                        return ' '.join(word.capitalize() for word in words)
                    else:
                        return name.capitalize()
            
            return domain.capitalize()
        except Exception as e:
            logging.warning(f"Error extracting publication name from URL {url}: {e}")
            return "Unknown source"
    
    def fetch_url_with_headers(self, url):
        """
        Fetch article content using requests with custom headers.
        
        Args:
            url (str): URL to fetch
        
        Returns:
            str: Article text or empty string
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Use newspaper3k for parsing
            article = Article(url)
            article.download(input_html=response.text)
            article.parse()
            
            return article.text or ""
        except Exception as e:
            logging.warning(f"Error fetching URL with headers: {e}")
            return ""

    def fetch_from_web_archive(self, url):
        """
        Fetch article content from Web Archive as a fallback.
        
        Args:
            url (str): Original URL
        
        Returns:
            str: Article text or empty string
        """
        try:
            # Encode the original URL
            encoded_url = urllib.parse.quote(url)
            archive_url = f"https://web.archive.org/web/{encoded_url}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(archive_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Use newspaper3k for parsing
            article = Article(url)
            article.download(input_html=response.text)
            article.parse()
            
            return article.text or ""
        except Exception as e:
            logging.warning(f"Error fetching from Web Archive: {e}")
            return ""

    def generic_text_extraction(self, url):
        """
        Perform generic text extraction as a last resort.
        
        Args:
            url (str): URL to extract text from
        
        Returns:
            str: Extracted text or empty string
        """
        try:
            # Use newspaper3k for generic extraction
            article = Article(url)
            article.download()
            article.parse()
            
            return article.text or ""
        except Exception as e:
            logging.warning(f"Generic text extraction failed: {e}")
            return ""

    def extract_article_text(self, url, timeout=10):
        """
        Extract full text from an article using newspaper3k.
        
        Args:
            url (str): URL of the article to extract
            timeout (int): Timeout for downloading article in seconds
        
        Returns:
            str: Extracted article text or empty string if extraction fails
        """
        try:
            # Configure newspaper
            config = Config()
            config.browser_user_agent = self.headers['User-Agent']
            config.request_timeout = timeout
            config.fetch_images = False
            
            # Check if content is protected/paywalled
            if self.is_protected_content(url):
                logging.info(f"Protected content detected for {url}, attempting archive services")
                # Try archive services first
                for service_url in self.archive_services:
                    content = self.try_archive_service(service_url, url)
                    if content:
                        return content
            
            # Skip video content
            if 'video' in url.lower() or '/video/' in url.lower():
                logging.info(f"Skipping video content from {url}")
                return ""
            
            # Try direct extraction first
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            if not article.text:
                # Fallback to custom extraction
                text = self.fetch_url_with_headers(url)
                if not text:
                    text = self.fetch_from_web_archive(url)
                if not text:
                    text = self.generic_text_extraction(url)
                return text
            
            return article.text.strip()
            
        except Exception as e:
            logging.error(f"Error extracting article text from {url}: {e}")
            # Try fallback methods
            try:
                text = self.fetch_url_with_headers(url)
                if not text:
                    text = self.fetch_from_web_archive(url)
                if not text:
                    text = self.generic_text_extraction(url)
                return text
            except Exception as e2:
                logging.error(f"All extraction methods failed for {url}: {e2}")
                return ""

    def extract_article_content(self, html_text):
        """Extract article content from HTML."""
        try:
            # Parse HTML
            soup = BeautifulSoup(html_text, 'html.parser')
            
            # Try various methods to extract article text
            article_text = ""
            
            # Try extracting text from specific article-related tags
            article_selectors = [
                'article', 
                'main', 
                '.article-body', 
                '.story-body', 
                '.post-content', 
                '#article-body', 
                '.content', 
                '.entry-content',
                '[role="article"]',
                '[role="main"]',
                '.post',
                '.story',
                '.article-content',
                '#content-body'
            ]
            
            # Remove unwanted elements
            for selector in ['.ad', '.advertisement', '.social-share', '.newsletter', '.sidebar', 'nav', 'header', 'footer']:
                for element in soup.select(selector):
                    element.decompose()
            
            # Try each selector
            for selector in article_selectors:
                content = soup.select_one(selector)
                if content:
                    # Extract text from paragraphs within the selected content
                    paragraphs = content.find_all('p')
                    article_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    
                    # If we found some text, break the loop
                    if article_text:
                        break
            
            # If no text found, fall back to all paragraphs
            if not article_text:
                paragraphs = soup.find_all('p')
                article_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            if not article_text:
                logger.warning("No article text found")
                return None, None
                
            # Clean up the text
            article_text = re.sub(r'\s+', ' ', article_text)  # Replace multiple spaces
            article_text = re.sub(r'\n\s*\n', '\n\n', article_text)  # Fix multiple newlines
            article_text = article_text.strip()
            
            # Truncate to a reasonable length
            article_text = article_text[:10000]
            
            return article_text, soup
            
        except Exception as e:
            logger.error(f"Error extracting article content: {str(e)}")
            return None, None

    def summarize_text(self, text, title=None, max_retries=3):
        """
        Generate a summary of the given text using Claude API.
        
        Args:
            text (str): The text to summarize
            title (str): Optional title for context
            max_retries (int): Maximum number of retries for API calls
            
        Returns:
            Dict: Summary data including the summary text and metadata
        """
        if not text:
            return {
                'headline': title or 'No headline available',
                'summary': 'No content available for summarization',
                'model': self.model,
                'timestamp': str(int(time.time()))
            }
            
        # Generate cache key
        cache_key = f"summary_{hashlib.md5(text.encode()).hexdigest()}"
        
        # Check cache first
        cached_summary = self.cache.get(cache_key)
        if cached_summary:
            return cached_summary
            
        # Retry loop for API calls
        for attempt in range(max_retries):
            try:
                # Make Claude API call with detailed error handling
                prompt = f"""Please provide a concise summary of the following text in 3-4 sentences. Focus on the key points and newsworthy information:

{text}

Return the summary in this JSON format:
{{
    "headline": "Brief headline that captures main point",
    "summary": "3-4 sentence summary of key points"
}}"""

                response = self.anthropic.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                summary_text = response.content[0].text.strip()
                
                # Try to parse as JSON
                try:
                    # Remove any potential markdown code block markers
                    summary_text = summary_text.replace('```json', '').replace('```', '').strip()
                    summary_dict = json.loads(summary_text)
                    
                    if isinstance(summary_dict, dict):
                        # Ensure required fields exist
                        summary_dict['headline'] = summary_dict.get('headline', title or 'No headline available')
                        summary_dict['summary'] = summary_dict.get('summary', 'Summary not available')
                        summary_dict['model'] = self.model
                        summary_dict['timestamp'] = str(int(time.time()))
                        
                        self.cache.set(cache_key, summary_dict)
                        return summary_dict
                        
                except json.JSONDecodeError:
                    # If JSON parsing fails, create a basic summary
                    summary_dict = {
                        'headline': title or 'No headline available',
                        'summary': summary_text,
                        'model': self.model,
                        'timestamp': str(int(time.time()))
                    }
                    self.cache.set(cache_key, summary_dict)
                    return summary_dict
                    
            except Exception as e:
                logger.error(f"Error in Claude API call (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        'headline': title or 'No headline available',
                        'summary': 'Failed to generate summary',
                        'model': 'Error',
                        'timestamp': str(int(time.time()))
                    }
                time.sleep(1)  # Wait before retrying
    
    @track_performance()
    def cluster_similar_articles(self, articles, similarity_threshold=0.5, max_clusters=10):
        """
        Advanced clustering of articles based on multi-dimensional semantic similarity.
        
        Uses sentence transformers to compute embeddings and cosine similarity
        to group related articles together. This helps reduce redundancy and
        provides a more comprehensive view of each news story.
        
        Args:
            articles (list): List of article dictionaries
            similarity_threshold (float): Threshold for clustering similar articles
            max_clusters (int): Maximum number of clusters to generate
        
        Returns:
            list: List of article clusters, where each cluster is a list of related articles
        """
        if not articles:
            logging.warning("No articles provided for clustering")
            return []
            
        try:
            # Extract titles and summaries for comparison
            texts = []
            for article in articles:
                title = article.get('title', '')
                summary_dict = article.get('summary', {})
                if not isinstance(summary_dict, dict):
                    summary_dict = {}
                summary = summary_dict.get('summary', '') if summary_dict else ''
                combined_text = f"{title} {summary}".strip()
                texts.append(combined_text)
                
            # Get embeddings using sentence transformer
            embeddings = self._get_sentence_transformer().encode(texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Initialize clusters
            clusters = []
            used_indices = set()  # Use a set for better performance
            
            # Create clusters based on similarity
            for i in range(len(articles)):
                if i in used_indices:
                    continue
                    
                # Start new cluster
                cluster = [articles[i]]
                used_indices.add(i)
                
                # Find similar articles
                for j in range(i + 1, len(articles)):
                    if j in used_indices:
                        continue
                        
                    if similarity_matrix[i][j] >= similarity_threshold:
                        cluster.append(articles[j])
                        used_indices.add(j)
                        
                clusters.append(cluster)
                
            # Sort clusters by size and limit to max_clusters
            clusters.sort(key=len, reverse=True)
            clusters = clusters[:max_clusters]
            
            # Add remaining articles as single-article clusters
            remaining_articles = [
                articles[i] for i in range(len(articles))
                if i not in used_indices
            ]
            
            for article in remaining_articles:
                clusters.append([article])
                
            return clusters
            
        except Exception as e:
            logging.error(f"Error in clustering articles: {str(e)}")
            logging.debug(traceback.format_exc())
            # Return each article as its own cluster
            return [[article] for article in articles]

    @track_performance()
    def process_feeds_with_clustering(self, feeds_file='rss_test.txt'):
        """
        Process RSS feeds with article clustering.
        
        Args:
            feeds_file (str): Path to file containing RSS feed URLs
        
        Returns:
            list: List of article clusters
        """
        try:
            # Process feeds
            articles = self.process_feeds(feeds_file)
            if not articles:
                return []
            
            # Cluster similar articles
            clusters = self.cluster_similar_articles(articles)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error in process_feeds_with_clustering: {str(e)}")
            return []

    def _process_cluster(self, cluster):
        """Process a single cluster and return its HTML content."""
        if len(cluster) == 1:
            return self._process_single_article(cluster[0])
        else:
            return self._process_multiple_articles(cluster)

    def _generate_html_footer(self):
        """Generate the HTML footer."""
        return f'''
            <div class="footer">
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} using Claude</p>
            </div>
        </body>
        </html>
        '''

    def _process_single_article(self, article):
        """Process a single article and return its HTML content."""
        summary_dict = article.get('summary', {})
        
        if isinstance(summary_dict, str):
            try:
                summary_dict = json.loads(summary_dict)
            except json.JSONDecodeError:
                summary_dict = {'headline': 'No Headline', 'summary': summary_dict}
        
        headline = str(summary_dict.get('headline', article.get('title', 'No headline available')))
        summary_text = str(summary_dict.get('summary', 'No summary available'))
        model = str(summary_dict.get('model', 'Unknown'))
        timestamp = summary_dict.get('timestamp')
        try:
            timestamp_int = int(timestamp)
            timestamp_str = datetime.fromtimestamp(timestamp_int).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            timestamp_str = 'Unknown'
            
        source = str(article.get('feed_source', 'Unknown Source'))
        link = str(article.get('link', '#'))
        
        return f'''
        <div class="article">
            <h2>{html.escape(headline)}</h2>
            <div class="article-meta">
                From <a href="{html.escape(link)}" target="_blank">{html.escape(source)}</a>
            </div>
            <div class="article-summary">{html.escape(summary_text)}</div>
            <div class="metadata">
                Model: {html.escape(model)} | Generated: {html.escape(timestamp_str)}
            </div>
        </div>
        '''

    def _process_multiple_articles(self, cluster):
        """Process multiple related articles and return their HTML content."""
        summary = self.generate_cluster_summary(cluster)
        if not summary:
            return ''
            
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                summary = {'headline': 'No Headline', 'summary': summary}
        
        headline = str(summary.get('headline', 'Related Articles'))
        summary_text = str(summary.get('summary', 'No summary available'))
        model = str(summary.get('model', 'Unknown'))
        timestamp = summary.get('timestamp')
        try:
            timestamp_int = int(timestamp)
            timestamp_str = datetime.fromtimestamp(timestamp_int).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            timestamp_str = 'Unknown'
        
        # Get sources from the cluster
        source_list = []
        for article in cluster:
            source_list.append({
                'title': article.get('title', 'No title'),
                'source': article.get('feed_source', 'Unknown Source'),
                'link': article.get('link', '#'),
                'pub_date': article.get('pub_date', '')
            })
        
        html_content = f'''
        <div class="article cluster">
            <h2>{html.escape(headline)}</h2>
            <div class="article-meta">
                From multiple sources:
                <ul class="source-list">
        '''
        
        for source in source_list:
            html_content += f'''
                <li>
                    <a href="{html.escape(source['link'])}" target="_blank">
                        {html.escape(source['title'])} ({html.escape(source['source'])})
                    </a>
                </li>
            '''
        
        html_content += f'''
                </ul>
            </div>
            <div class="article-summary">{html.escape(summary_text)}</div>
            <div class="metadata">
                Model: {html.escape(model)} | Generated: {html.escape(timestamp_str)}
            </div>
        </div>
        '''
        
        return html_content

    def generate_cluster_summary(self, cluster):
        """
        Generate a summary for a cluster of related articles.
        
        Args:
            cluster (list): List of related articles
            
        Returns:
            dict: Summary data including headline, summary text, and metadata
        """
        try:
            # Get the most recent article's title as the main headline
            main_article = max(cluster, key=lambda x: x.get('published', ''))
            headline = main_article.get('title', 'No headline available')
            
            # Combine summaries from all articles
            summaries = []
            for article in cluster:
                summary_dict = article.get('summary', {})
                if isinstance(summary_dict, str):
                    try:
                        summary_dict = json.loads(summary_dict)
                    except:
                        summary_dict = {'summary': summary_dict}
                
                if summary_dict and 'summary' in summary_dict:
                    summaries.append(summary_dict['summary'])
            
            combined_text = " ".join(summaries)
            
            if not combined_text:
                return {
                    'headline': headline,
                    'summary': "No summary available for this cluster",
                    'model': 'Fallback',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Generate a new summary from the combined text
            try:
                cluster_summary = self.summarize_text(combined_text, headline)
                if isinstance(cluster_summary, str):
                    cluster_summary = {
                        'headline': headline,
                        'summary': cluster_summary,
                        'model': 'Claude',
                        'timestamp': datetime.now().isoformat()
                    }
                return cluster_summary
            except Exception as e:
                logging.error(f"Error generating cluster summary: {str(e)}")
                return {
                    'headline': headline,
                    'summary': "Failed to generate cluster summary",
                    'model': 'Error',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logging.error(f"Error in generate_cluster_summary: {str(e)}")
            return {
                'headline': "Cluster Summary Error",
                'summary': "Failed to process cluster summary",
                'model': 'Error',
                'timestamp': datetime.now().isoformat()
            }

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
    Main function that orchestrates the RSS reader workflow.
    
    This function:
    1. Sets up logging and configuration
    2. Initializes the RSS reader
    3. Processes feeds and generates summaries
    4. Clusters similar articles
    5. Generates HTML output
    
    The function handles errors gracefully and provides
    informative error messages if something goes wrong.
    """
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Initialize RSS reader
        logging.info("🚀 Initializing RSS Reader...")
        reader = RSSReader()
        
        # Process feeds with clustering
        logging.info("📊 Processing feeds and clustering articles...")
        clusters = reader.process_feeds_with_clustering()
        
        if not clusters:
            logging.warning("⚠️ No articles found or processed")
            return
        
        # Generate HTML output
        logging.info("📝 Generating HTML output...")
        output_file = reader.generate_html_output(clusters)
        
        if output_file:
            # Open in browser
            logging.info(f"✨ Opening {output_file} in browser...")
            webbrowser.open(output_file)
        
    except Exception as e:
        logging.error(f"❌ Error in main: {str(e)}")
        logging.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Test Readwise token loading
    test_readwise_token()
    
    # Run main function
    main()
