# AI-Powered RSS Reader

A modern RSS feed reader that uses AI to generate concise summaries of articles and intelligently clusters related content.

## Features

- **AI-Powered Summaries**: Uses Claude API to generate concise, informative summaries of articles
- **Smart Article Clustering**: Groups similar articles together and provides unified summaries
- **Multiple Sources**: Displays original sources for each article or cluster
- **Clean Interface**: Modern, responsive design with clear typography and visual hierarchy
- **Caching**: Efficient caching system for summaries to avoid redundant API calls
- **Fallback Options**: Uses HuggingFace transformers as a fallback for summarization

## Requirements

- Python 3.8+
- Anthropic API key (for Claude)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Step 1: Generate RSS Feed Summaries

First, run the RSS reader to fetch and summarize articles:
```bash
python3 rss_reader.py
```

This will:
1. Fetch articles from configured RSS feeds
2. Generate AI-powered summaries using Claude
3. Cluster similar articles together
4. Create an HTML summary in `output/rss_summary_[timestamp].html`

### Step 2: View Curated Results

To view the generated summaries in a clean web interface:
```bash
python3 webserver.py
```

Then:
1. Open your browser and navigate to `http://localhost:5000`
2. Browse through the curated article summaries
3. Use the agenda navigation to explore article clusters
4. Toggle between light and dark themes using the theme button
5. Click article titles to read the full content

Note: Run `rss_reader.py` whenever you want to fetch and summarize new articles, then refresh your browser to see the updates.

## Configuration

Edit `rss_reader.py` to configure:
- RSS feed URLs
- Batch size for processing
- Clustering parameters
- Cache settings
- Output formatting

## Features in Detail

### AI Summaries
- Concise, factual summaries using Claude API
- Automatic headline extraction
- Source attribution and timestamps
- Fallback to HuggingFace transformers if needed

### Article Clustering
- Semantic similarity-based clustering
- Combined summaries for related articles
- Links to all original sources
- Visual distinction for clustered content

### User Interface
- Clean, modern design
- Mobile-responsive layout
- Easy navigation
- Clear source attribution
- Timestamp and model information

## Contributing

Feel free to open issues or submit pull requests for:
- New features
- Bug fixes
- Documentation improvements
- UI enhancements

## License

MIT License - feel free to use and modify as needed.
