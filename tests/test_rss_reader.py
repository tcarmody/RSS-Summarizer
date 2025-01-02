import pytest
from rss_reader import RSSReader
from unittest.mock import Mock, patch
import json
from datetime import datetime

@pytest.fixture
def rss_reader():
    return RSSReader()

@pytest.fixture
def sample_cluster():
    return [
        {
            'title': 'Test Article 1',
            'summary': 'This is test content for article 1',
            'feed_source': 'Test Source 1',
            'link': 'http://test1.com',
            'pub_date': 'Wed, 02 Jan 2025 15:04:22 GMT',
            'published_parsed': (2025, 1, 2, 15, 4, 22, 0, 0, 0)
        },
        {
            'title': 'Test Article 2',
            'summary': 'This is test content for article 2',
            'feed_source': 'Test Source 2',
            'link': 'http://test2.com',
            'pub_date': 'Wed, 02 Jan 2025 14:04:22 GMT',
            'published_parsed': (2025, 1, 2, 14, 4, 22, 0, 0, 0)
        }
    ]

def test_process_multiple_articles_basic(rss_reader, sample_cluster):
    with patch.object(rss_reader, 'generate_cluster_summary') as mock_summary:
        mock_summary.return_value = {
            'headline': 'Test Headline',
            'summary': 'Test summary content',
            'model': 'claude-3',
            'timestamp': 1704229462  # 2025-01-02 15:04:22
        }
        
        result = rss_reader._process_multiple_articles(sample_cluster)
        
        assert 'Test Headline' in result
        assert 'Test summary content' in result
        assert 'claude-3' in result
        assert 'Test Source 1' in result
        assert 'Test Source 2' in result
        assert 'http://test1.com' in result
        assert 'http://test2.com' in result

def test_generate_cluster_summary_success(rss_reader, sample_cluster):
    mock_response = Mock()
    mock_response.content = [{'text': 'Test generated summary'}]
    
    with patch.object(rss_reader, 'anthropic') as mock_anthropic:
        mock_anthropic.messages.create.return_value = mock_response
        
        result = rss_reader.generate_cluster_summary(sample_cluster)
        
        assert isinstance(result, dict)
        assert result['headline'] == 'Test Article 1'  # Should use most recent article's title
        assert result['summary'] == 'Test generated summary'
        assert result['model'] == rss_reader.model
        assert isinstance(result['timestamp'], int)

def test_generate_cluster_summary_error_handling(rss_reader, sample_cluster):
    with patch.object(rss_reader, 'anthropic') as mock_anthropic:
        mock_anthropic.messages.create.side_effect = Exception('API Error')
        
        result = rss_reader.generate_cluster_summary(sample_cluster)
        
        assert isinstance(result, dict)
        assert 'Error' in result['model']
        assert 'Error generating summary' in result['summary']
        assert isinstance(result['timestamp'], int)

def test_empty_cluster_handling(rss_reader):
    result = rss_reader.generate_cluster_summary([])
    
    assert isinstance(result, dict)
    assert 'Error Processing Cluster' in result['headline']
    assert 'Failed to process' in result['summary']
    assert isinstance(result['timestamp'], int)
