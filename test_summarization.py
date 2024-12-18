from rss_reader import RSSReader
import os

def test_summarization():
    # Create RSSReader instance
    reader = RSSReader()
    
    # Test article text
    test_article = """
    The James Webb Space Telescope has made an extraordinary discovery in deep space. 
    Scientists have identified a galaxy that appears to be more than 13 billion years old, 
    potentially one of the earliest formed after the Big Bang. This finding challenges 
    some of our existing theories about galaxy formation in the early universe.
    The telescope's infrared capabilities allowed it to peer through cosmic dust and capture 
    images of this ancient galaxy with unprecedented clarity. The research team spent months 
    analyzing the data to confirm their findings. This discovery provides valuable insights 
    into how the first galaxies formed and evolved in the early universe.
    The findings have been published in a peer-reviewed journal and have generated significant 
    interest in the astronomical community. Further observations are planned to study this 
    galaxy in more detail and search for other similar ancient structures.
    """
    
    print("Testing article summarization...")
    try:
        summary_result = reader.summarize_text(test_article)
        print("\nSummary:")
        print(summary_result.get('summary', 'No summary generated'))
        
        print("\nGenerated Headline:")
        print(summary_result.get('headline', 'No headline generated'))
        
        print("\nModel Used:")
        print(summary_result.get('model', 'Unknown model'))
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_summarization()
    print(f"\nTest completed {'successfully' if success else 'with errors'}")
