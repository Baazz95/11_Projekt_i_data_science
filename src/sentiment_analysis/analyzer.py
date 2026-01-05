"""
Sentiment Analysis Module
Provides multiple sentiment analysis methods: VADER, RoBERTa, and TweetEval.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Main sentiment analysis class supporting multiple models."""
    
    def __init__(self, model_type: str = "vader"):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_type: Type of model to use ("vader", "roberta")
        """
        self.model_type = model_type.lower()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the specified sentiment analysis model."""
        if self.model_type == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.model = SentimentIntensityAnalyzer()
            print("Loaded VADER sentiment analyzer")
            
        elif self.model_type == "roberta":
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                from transformers import pipeline
                
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                self.model = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=-1  # Use CPU, set to 0 for GPU
                )
                print("Loaded RoBERTa sentiment analyzer")
            except Exception as e:
                print(f"Error loading RoBERTa model: {e}")
                print("Falling back to VADER")
                self.model_type = "vader"
                self._load_model()
                
        elif self.model_type == "tweeteval":
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                from transformers import pipeline
                
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                self.model = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=-1
                )
                print("Loaded TweetEval sentiment analyzer")
            except Exception as e:
                print(f"Error loading TweetEval model: {e}")
                print("Falling back to VADER")
                self.model_type = "vader"
                self._load_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores and label
        """
        if pd.isna(text) or text == "":
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0
            }
        
        if self.model_type == "vader":
            scores = self.model.polarity_scores(text)
            
            # Determine label based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            
            return {
                "sentiment_score": compound,
                "sentiment_label": label,
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu']
            }
        
        else:  # RoBERTa or TweetEval
            try:
                result = self.model(text, truncation=True, max_length=512)[0]
                label = result['label'].lower()
                score = result['score']
                
                # Map labels to standard format
                if 'positive' in label or 'pos' in label:
                    sentiment_score = score
                    sentiment_label = "positive"
                elif 'negative' in label or 'neg' in label:
                    sentiment_score = -score
                    sentiment_label = "negative"
                else:
                    sentiment_score = 0.0
                    sentiment_label = "neutral"
                
                return {
                    "sentiment_score": sentiment_score,
                    "sentiment_label": sentiment_label,
                    "positive": score if sentiment_label == "positive" else 0.0,
                    "negative": score if sentiment_label == "negative" else 0.0,
                    "neutral": score if sentiment_label == "neutral" else 0.0
                }
            except Exception as e:
                print(f"Error analyzing text: {e}")
                return {
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral",
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0
                }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "full_text") -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame containing texts to analyze
            text_column: Name of the column containing text
        
        Returns:
            DataFrame with added sentiment columns
        """
        df = df.copy()
        
        # Initialize sentiment columns
        df['sentiment_score'] = 0.0
        df['sentiment_label'] = 'neutral'
        df['sentiment_positive'] = 0.0
        df['sentiment_negative'] = 0.0
        df['sentiment_neutral'] = 0.0
        
        # Analyze each text
        print(f"Analyzing sentiment for {len(df)} texts using {self.model_type.upper()}...")
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
            text = row[text_column]
            sentiment_result = self.analyze_text(str(text))
            results.append(sentiment_result)
        
        # Add results to dataframe
        sentiment_df = pd.DataFrame(results)
        df['sentiment_score'] = sentiment_df['sentiment_score']
        df['sentiment_label'] = sentiment_df['sentiment_label']
        df['sentiment_positive'] = sentiment_df['positive']
        df['sentiment_negative'] = sentiment_df['negative']
        df['sentiment_neutral'] = sentiment_df['neutral']
        
        return df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of sentiment analysis.
        
        Args:
            df: DataFrame with sentiment analysis results
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "total_texts": len(df),
            "mean_sentiment_score": df['sentiment_score'].mean(),
            "median_sentiment_score": df['sentiment_score'].median(),
            "std_sentiment_score": df['sentiment_score'].std(),
            "positive_count": (df['sentiment_label'] == 'positive').sum(),
            "negative_count": (df['sentiment_label'] == 'negative').sum(),
            "neutral_count": (df['sentiment_label'] == 'neutral').sum(),
            "positive_percentage": (df['sentiment_label'] == 'positive').mean() * 100,
            "negative_percentage": (df['sentiment_label'] == 'negative').mean() * 100,
            "neutral_percentage": (df['sentiment_label'] == 'neutral').mean() * 100
        }
        
        return summary

