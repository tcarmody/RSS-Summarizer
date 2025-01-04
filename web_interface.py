from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from rss_reader import FavoritesManager, ExportManager, RSSReaderConfig, RSSReader
import os
from werkzeug.utils import secure_filename
import json
import hashlib
import feedparser
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s: %(message)s')

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize managers and reader
favorites_manager = FavoritesManager()
export_manager = ExportManager()
rss_reader = RSSReader()

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

        # Check for API key before proceeding
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logging.error("ANTHROPIC_API_KEY not found in environment variables")
                raise ValueError("ANTHROPIC_API_KEY not set. Please set this environment variable to enable article summarization.")
        except Exception as e:
            logging.error(f"Error with API key: {str(e)}")
            return [{'title': 'Configuration Error', 
                    'url': '#',
                    'summary': 'Please set the ANTHROPIC_API_KEY environment variable to enable article summarization.',
                    'id': 'config_error'}]

        with open(feed_path, 'r') as f:
            feed_urls = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
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
                    
                    # Create article dictionary
                    article = {
                        'title': entry.get('title', 'No Title'),
                        'url': entry.get('link', '#'),
                        'content': content,
                        'published': entry.get('published', datetime.now().isoformat()),
                    }
                    
                    # Generate summary
                    try:
                        summary_prompt = """You are an expert at creating concise, informative summaries of articles. Your task is to summarize the following article:
<article>
{}
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
4. Tone: Factual, informative, and free from exaggeration, hype, or marketing speak
5. Headline:
   - Create a headline in sentence case
   - Avoid repeating too many words or phrases from the summary

Format your response exactly like this, with a headline followed by the summary on a new line:
[Headline]
[Summary]"""

                        response = rss_reader.client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=300,
                            temperature=0.7,
                            system=summary_prompt.format(content),
                            messages=[{
                                "role": "user",
                                "content": "Please provide the summary following the guidelines above."
                            }]
                        )
                        
                        # Split response into headline and summary
                        text = response.content[0].text.strip()
                        lines = text.split('\n', 1)
                        if len(lines) == 2:
                            article['headline'] = lines[0].strip()
                            article['summary'] = lines[1].strip()
                        else:
                            article['headline'] = article['title']
                            article['summary'] = text
                            
                    except Exception as e:
                        logging.error(f"Error summarizing article: {str(e)}")
                        article['headline'] = article['title']
                        article['summary'] = "Summary not available."
                    
                    # Generate tags
                    try:
                        article['tags'] = rss_reader.generate_tags(content)
                    except Exception as e:
                        logging.error(f"Error generating tags: {str(e)}")
                        article['tags'] = []
                    
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
    
    # Get all favorites and their tags
    favorites = favorites_manager.get_favorites()
    
    # Get all available tags with their counts
    tags = favorites_manager.get_tags()
    
    return render_template('index.html', 
                         current_articles=current_articles,
                         favorites=favorites,
                         tags=tags)

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
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'articles_{timestamp}.{format}'
        output_file = os.path.join(output_dir, filename)
        
        # Export the articles
        favorites_manager.export_favorites(
            output_file=output_file,
            format=format,
            tag=tag,
            include_summary=include_summary
        )
        
        # For HTML format, we can serve the file directly
        if format == 'html':
            return jsonify({
                'status': 'success',
                'message': f'Articles exported successfully! <a href="/output/{filename}" target="_blank">View Export</a>'
            })
        else:
            return jsonify({
                'status': 'success',
                'message': f'Articles exported successfully to {filename}'
            })
    except Exception as e:
        logging.error(f"Export error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Export failed: {str(e)}'
        })

@app.route('/output/<path:filename>')
def serve_export(filename):
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    return send_from_directory(output_dir, filename)

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
