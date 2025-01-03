from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from rss_reader import FavoritesManager, ExportManager, TagGenerator, RSSReaderConfig, ArticleSummarizer
import os
from werkzeug.utils import secure_filename
import json
import hashlib
import feedparser
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s: %(message)s')

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize managers and summarizer
favorites_manager = FavoritesManager()
export_manager = ExportManager()
tag_generator = TagGenerator()
summarizer = ArticleSummarizer()

# Load RSS feed data
def load_rss_feeds():
    feeds = []
    try:
        logging.info("Starting to load RSS feeds...")
        feed_path = os.path.join(os.path.dirname(__file__), 'rss_test.txt')
        logging.info(f"Reading feeds from: {feed_path}")
        
        if not os.path.exists(feed_path):
            logging.error(f"RSS feeds file not found at: {feed_path}")
            return feeds

        with open(feed_path, 'r') as f:
            # Strip comments and whitespace from URLs
            feed_urls = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove any trailing comments
                    url = line.split('#')[0].strip()
                    feed_urls.append(url)
                    
            logging.info(f"Found {len(feed_urls)} feed URLs")
            
        for url in feed_urls:
            try:
                logging.info(f"Parsing feed: {url}")
                feed = feedparser.parse(url)
                
                if hasattr(feed, 'status') and feed.status != 200:
                    logging.error(f"Error fetching feed {url}: Status {feed.status}")
                    continue
                    
                articles = []
                for entry in feed.entries[:10]:  # Get latest 10 articles
                    # Get the full content or summary
                    content = entry.get('content', [{'value': ''}])[0].get('value', '') or entry.get('summary', '')
                    
                    # Generate summary using our summarizer
                    try:
                        summary = summarizer.summarize_article(content)
                    except Exception as e:
                        logging.error(f"Error summarizing article: {str(e)}")
                        summary = "Summary not available."
                    
                    article = {
                        'title': entry.get('title', ''),
                        'url': entry.get('link', ''),
                        'summary': summary,
                        'published': entry.get('published', datetime.now().isoformat()),
                    }
                    article['id'] = generate_article_id(article)
                    articles.append(article)
                logging.info(f"Successfully parsed {len(articles)} articles from {url}")
                feeds.extend(articles)
            except Exception as e:
                logging.error(f"Error parsing feed {url}: {str(e)}")
        
        logging.info(f"Total articles loaded: {len(feeds)}")
    except Exception as e:
        logging.error(f"Error loading RSS feeds: {str(e)}")
    return feeds

def generate_article_id(article):
    """Generate a unique ID for an article based on its URL and title"""
    unique_string = f"{article['url']}{article['title']}"
    return hashlib.md5(unique_string.encode()).hexdigest()

@app.route('/')
def index():
    # Get current RSS feed articles
    current_articles = load_rss_feeds()
    
    # Get all favorites and available tags
    favorites = favorites_manager.get_favorites()
    all_tags = favorites_manager.get_tags()
    
    return render_template('index.html', 
                         current_articles=current_articles,
                         favorites=favorites, 
                         tags=all_tags)

@app.route('/favorite/<article_id>')
def toggle_favorite(article_id):
    try:
        if article_id in favorites_manager.favorites:
            favorites_manager.remove_favorite(article_id)
            flash('Article removed from favorites', 'success')
        else:
            article = get_article_by_id(article_id)
            if article:
                favorites_manager.add_favorite(article)
                flash('Article added to favorites', 'success')
            else:
                flash('Article not found', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/add_tag', methods=['POST'])
def add_tag():
    article_id = request.form.get('article_id')
    tags = request.form.get('tags').split(',')
    try:
        favorites_manager.add_tags(article_id, tags)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/remove_tag', methods=['POST'])
def remove_tag():
    article_id = request.form.get('article_id')
    tag = request.form.get('tag')
    try:
        favorites_manager.remove_tag(article_id, tag)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/export', methods=['POST'])
def export():
    format = request.form.get('format', 'html')
    tag = request.form.get('tag')
    include_summary = request.form.get('include_summary', 'true') == 'true'
    
    try:
        output_file = f'output/exported_articles.{format}'
        favorites_manager.export_favorites(
            output_file=output_file,
            format=format,
            tag=tag,
            include_summary=include_summary
        )
        return jsonify({
            'status': 'success',
            'message': f'Articles exported successfully to {output_file}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/search')
def search():
    query = request.args.get('q', '')
    results = favorites_manager.search_favorites(query) if query else []
    return jsonify({'results': results})

def get_article_by_id(article_id):
    """Retrieve article data from either current RSS feed or favorites"""
    # First check favorites
    if article_id in favorites_manager.favorites:
        return favorites_manager.favorites[article_id]
    
    # Then check current RSS feed articles
    current_articles = load_rss_feeds()
    for article in current_articles:
        if article['id'] == article_id:
            return article
    return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
