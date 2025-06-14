"""

##############################################
Scoring System
##############################################

Window associated metrics:
    - Word rate per time window.
    - Average pitch in time window. 
    - Average gap between words in time window.
    - Number of pauses. 
    - Variation in pitch in time window.
    - Number of filler words. 
    - Lexical diversity in time window.

Full transcript metrics:
   - Full Transcript.
   - Lexical diversity = Repeatability
   - Readability 

##############################################
##############################################
   
Improvements:


1) Add in fixed tests. 
2) Add in automatic dependency installation & checking.
3) Check deployment


##############################################
##############################################
"""

import argparse
import ffmpeg
import whisper
import parselmouth
import pandas as pd
import numpy as np
import textstat
from nltk import word_tokenize
from transformers import pipeline
from collections import defaultdict
import os
import webbrowser

# Ensure nltk punkt tokenizer is available
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


###############################################################################
#                              Core Functions                                 #
###############################################################################

def convert_m4a_to_wav(input_file, output_file):
    """
    Converts an .m4a audio file to a .wav file using the ffmpeg Python library.

    Args:
        input_file (str): Path to the input .m4a file.
        output_file (str): Path to the output .wav file.
    """

    try:
        ffmpeg.input(input_file).output(output_file).run(overwrite_output=True, quiet=True)
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print(f"Error during conversion: {e.stderr.decode()}")


def get_wav_length(wav_file):
    """
    Get the length of a .wav file in seconds using ffmpeg.

    Args:
        wav_file (str): Path to the .wav file.

    Returns:
        float: Length of the .wav file in seconds.
    """
    try:
        probe = ffmpeg.probe(wav_file)
        duration = float(probe['format']['duration'])
        return duration
    except ffmpeg.Error as e:
        print(f"Error getting file duration: {e.stderr.decode()}")
        return None



def WhisperCreateTranscript(input_file,
                            Model='medium',
                            Language='en'):
    """
    Transcribes an audio file using the Whisper model.
    Args:
        input_file (str): Path to the input audio file.
        Model (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        Language (str): Language of the audio file.
    Returns:
        Whisper Output Dictionary.
    """

    # Load the Whisper model and transcribe the audio
    model = whisper.load_model(Model)
    result = model.transcribe(input_file, fp16=False, word_timestamps=True,language=Language)

    return result

 
def extract_word_pitch_data(output_wav, 
                            word_timestamps):
    """
    Extracts pitch data (mean, max, min) for each word in the audio file.

    Args:
        output_wav (str): Path to the .wav audio file.
        word_timestamps (list): List of word timestamps with start and end times.

    Returns:
        pd.DataFrame: DataFrame containing word pitch data (mean, max, min).
    """
    snd = parselmouth.Sound(output_wav)

    # Remove pitch ceiling and floor by setting extreme values
    pitch = snd.to_pitch()

    # Extract time and pitch values
    pitch_values = pitch.selected_array['frequency']  # Extract pitch frequencies
    pitch_times = pitch.xs()  # Extract corresponding time points

    # Filter out unvoiced frames (pitch values of 0)
    pitch_values[pitch_values == 0] = np.nan  # Replace unvoiced frames with NaN

    # Subset pitch times and values for each word
    word_pitch_data = []

    for word_info in word_timestamps:
        word_start = word_info['start']
        word_end = word_info['end']
        
        # Find indices of pitch times within the word's time range
        indices = (pitch_times >= word_start) & (pitch_times <= word_end)
        
        # Extract corresponding pitch values
        word_pitch_values = pitch_values[indices]
        
        if word_pitch_values.size == 0 or np.all(np.isnan(word_pitch_values)):  # Handle empty or all-NaN pitch values
            mean_pitch = np.nan
            max_pitch = np.nan
            min_pitch = np.nan
        else:
            # Calculate mean, max, and min pitch values, ignoring NaN
            mean_pitch = np.nanmean(word_pitch_values)
            max_pitch = np.nanmax(word_pitch_values)
            min_pitch = np.nanmin(word_pitch_values)
        
        # Store the word and its pitch data
        word_pitch_data.append({
            'word': word_info['word'],
            'start': word_start,
            'end': word_end,
            'mean_pitch': mean_pitch,
            'max_pitch': max_pitch,
            'min_pitch': min_pitch
        })

    # Convert the list of dictionaries to a pandas DataFrame
    word_pitch_df = pd.DataFrame(word_pitch_data)

    # Calculate the standard deviation of the mean pitch
    PitchStd = word_pitch_df['mean_pitch'].std()

    # Calculate the overall mean of the mean pitch
    PitchMean = word_pitch_df['mean_pitch'].mean()

    return word_pitch_df, PitchStd, PitchMean



def extract_pauses(word_timestamps, pause_threshold=0.5):
    """
    Extracts the time gaps between words and identifies pauses in the speech.

    Args:
        word_timestamps (list): List of word timestamps with start and end times.
        pause_threshold (float): Minimum duration (in seconds) to consider a gap as a pause.

    Returns:
        list: A list of dictionaries containing word pairs and their time gaps.
        list: A list of pauses where the gap exceeds the pause threshold.
    """
    time_gaps = []
    pauses = []

    for i in range(len(word_timestamps) - 1):
        current_word = word_timestamps[i]
        next_word = word_timestamps[i + 1]

        # Calculate the time gap between the current word and the next word
        gap = next_word['start'] - current_word['end']


        CommaOrPeriod = False
        if current_word['word'][-1] in [',', '.', '!', '?']:
            CommaOrPeriod = True

        time_gaps.append({
            'word_1': current_word['word'],
            'word_2': next_word['word'],
            'CommaOrPeriod': CommaOrPeriod,
            'gap': gap
        })

    # Convert to dataframe
    time_gaps = pd.DataFrame(time_gaps)

    # Check if the gap exceeds the pause threshold and count number of pauses
    pauses = len(time_gaps[time_gaps['gap'] > pause_threshold])

    # Calculate the mean time gap between words
    mean_time_gap = np.mean([time_gaps['gap']])

    return mean_time_gap, time_gaps, pauses



def lexical_diversity(text):
    """
    Calculate the lexical diversity of a given text.
    Lexical diversity is defined as the ratio of unique words (types) to the total number of words (tokens).
    Args:
        text (str): Input text to analyze.
    Returns:
        float: Lexical diversity ratio (types/tokens).
    """
    tokens = word_tokenize(text)
    types = set(tokens)
    return len(types) / len(tokens)


def detect_filler_words(transcript, filler_words=None):
    """
    Detects filler words in a given transcript and calculates their total count.

    Args:
        transcript (str): The transcript text to analyze.
        filler_words (list): A list of filler words to detect. Defaults to common English filler words.

    Returns:
        dict: A dictionary with filler words as keys and their counts as values.
        int: Total count of all filler words detected.
    """
    if filler_words is None:
        filler_words = [
            "um", "uh", "like", "you know", "so", "well", "hmm", "er", "err", "ah", "okay",
            "actually", "basically", "literally", "right", "I mean", "sort of", "kind of",
            "you see", "I guess", "you know what I mean", "alright", "anyway", "just",
            "seriously", "honestly", "no way", "for real", "gotcha", "y'know",
            "I dunno", "meh", "whoa", "like I said", "kinda", "sorta", "uh-huh"
        ]

    # Tokenize the transcript into words
    tokens = word_tokenize(transcript.lower())

    # Count occurrences of each filler word
    filler_word_counts = {word: tokens.count(word) for word in filler_words if word in tokens}

    # Calculate the total count of all filler words
    total_filler_count = sum(filler_word_counts.values())

    return filler_word_counts, total_filler_count



def readability_grade(text):
    """
    Calculate the Flesch-Kincaid Grade Level of a given text.
    Args:
        text (str): Input text to analyze.
    Returns:
        float: Flesch-Kincaid Grade Level.
    """
    return textstat.flesch_kincaid_grade(text)


def analyze_sentiment_transformers(transcript):
    """
    Analyze sentiment using a Hugging Face transformer model.

    Args:
        transcript (str): The transcript text.

    Returns:
        dict: Sentiment label and score.
    """
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_result = sentiment_pipeline(transcript)
    return sentiment_result  # Return the first result (label and score)


def analyze_emotions(transcript):
    """
    Analyze emotions using a Hugging Face transformer model.

    Args:
        transcript (str): The transcript text.

    Returns:
        dict: Emotion labels and scores.
    """
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    emotion_result = emotion_pipeline(transcript)
    return emotion_result


def Zero_Shot_Analyse_Emotions(transcript):
    # Load the BART large MNLI model
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Define your emotion candidate labels
    '''
    candidate_labels = [
        # Positive Emotions
        "Pride","Hope", "Enthusiasm",

        # Negative Emotions
        "Anxiety", "Frustration", "Hurt",

        # Neutral or Ambiguous Emotions
        "Skepticism","Thoughtfulness", "Uncertainty",

        # Meta-labels
        "Empathetic", "Sarcastic", "Passive-Aggressive", "Hostile", "Calm",
        "Composed", "Assertive", "Defeated","Confident"
    ]
    '''
    candidate_labels = ['confident','unsure','neutral']

    # Run classification
    result = classifier(transcript, candidate_labels, multi_label=True)

    # Print results sorted by score
    sorted_result = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)

    return sorted_result



def score_metrics(df,ID):
    
    # Define scoring functions
    def score_wpm(wpm):
        if 120 <= wpm <= 150:
            return 5
        elif 100 <= wpm < 120 or 150 < wpm <= 170:
            return 4
        elif 90 <= wpm < 100 or 170 < wpm <= 180:
            return 3
        elif 80 <= wpm < 90 or 180 < wpm <= 200:
            return 2
        else:
            return 1

    def score_pitch_std(std):
        if 30 <= std <= 60:
            return 5
        elif 20 <= std < 30 or 60 < std <= 70:
            return 4
        elif 15 <= std < 20 or 70 < std <= 80:
            return 3
        elif 10 <= std < 15 or 80 < std <= 90:
            return 2
        else:
            return 1

    def score_pause(pause):
        if 0.1 <= pause <= 0.3:
            return 5
        elif 0.05 <= pause < 0.1 or 0.3 < pause <= 0.5:
            return 4
        elif pause < 0.05 or 0.5 < pause <= 0.7:
            return 3
        else:
            return 2 if pause <= 1.0 else 1

    def score_fillers(count):
        if count == 0:
            return 5
        elif count <= 2:
            return 4
        elif count <= 5:
            return 3
        elif count <= 10:
            return 2
        else:
            return 1

    def score_lexical_diversity(ld):
        if ld >= 0.6:
            return 5
        elif ld >= 0.5:
            return 4
        elif ld >= 0.4:
            return 3
        elif ld >= 0.3:
            return 2
        else:
            return 1

    def score_readability(grade):
        ideal = 8
        diff = abs(grade - ideal)
        if diff <= 1:
            return 5
        elif diff <= 2:
            return 4
        elif diff <= 3:
            return 3
        elif diff <= 4:
            return 2
        else:
            return 1

    # Define the metrics to score
    metrics_to_score = {
        "Average_Words_Per_Minute": score_wpm,
        "Mean_Pitch_Std": score_pitch_std,
        "Mean_Pause": score_pause,
        "Total_Filler_Count": score_fillers,
        "Overall_Lexical_Diversity": score_lexical_diversity,
        "Overall_Readability": score_readability,
    }

    # Initialize the "Score" column with NaN
    df["Score"] = np.nan

    # Calculate scores for each metric if applicable
    for metric, scoring_function in metrics_to_score.items():
        if metric in df.index:
            df.at[metric, "Score"] = scoring_function(df.loc[metric, ID])

    return df



def create_html_report(SummaryDf, WindowsMetricDf, ID, output_dir):
    """
    Creates an HTML report with color-coded summary table and time window plots.
    
    Args:
        SummaryDf (pd.DataFrame): Summary metrics dataframe
        WindowsMetricDf (pd.DataFrame): Window-level metrics dataframe
        ID (str): Audio file identifier
        output_dir (str): Output directory path
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import base64
    from io import BytesIO
    
    # Define color mapping for scores
    def get_score_color(score):
        if pd.isna(score):
            return '#f8f9fa'  # Light gray for no score
        elif score == 5:
            return '#28a745'  # Green
        elif score == 4:
            return '#6f42c1'  # Purple
        elif score == 3:
            return '#ffc107'  # Yellow
        elif score == 2:
            return '#fd7e14'  # Orange
        elif score == 1:
            return '#dc3545'  # Red
        else:
            return '#f8f9fa'
    
    # Define ideal/target values for metrics
    def get_ideal_value(metric):
        ideal_values = {
            "Average_Words_Per_Minute": "120-150",
            "Mean_Pitch_Std": "30-60 Hz",
            "Mean_Pause": "0.1-0.3 s",
            "Total_Filler_Count": "0-2",
            "Overall_Lexical_Diversity": "≥0.6",
            "Overall_Readability": "7-9 (Grade Level)",
            "Duration": "N/A",
            "Word Count": "N/A",
            "Mean_Pitch": "N/A",
            "Mean_Pause_Std": "N/A",
            "ID": "N/A",
            "Date": "N/A",
            "User_Label": "N/A"
        }
        return ideal_values.get(metric, "N/A")
    
    # Separate metrics into core performance and emotional analysis
    def is_emotional_metric(metric):
        return any(keyword in metric for keyword in [
            "DistilBERT", "DistilRoBERTa", "BART", "Emotion", "Sentiment"
        ])
    
    # Split the dataframe
    core_metrics = SummaryDf[~SummaryDf.index.map(is_emotional_metric)]
    emotional_metrics = SummaryDf[SummaryDf.index.map(is_emotional_metric)]
    
    # Get the transcript from the global variable
    transcript_text = globals().get('Transcript', 'Transcript not available')
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Converser Analysis Report - {ID}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #007bff;
                margin-top: 30px;
            }}
            .transcript-container {{
                margin: 20px 0;
                border: 2px solid #007bff;
                border-radius: 5px;
                background-color: #f8f9fa;
            }}
            .transcript-header {{
                background-color: #007bff;
                color: white;
                padding: 10px 15px;
                margin: 0;
                font-size: 18px;
                font-weight: bold;
            }}
            .transcript-box {{
                height: 300px;
                overflow-y: auto;
                padding: 15px;
                background-color: white;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
                border: none;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .transcript-box::-webkit-scrollbar {{
                width: 12px;
            }}
            .transcript-box::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 6px;
            }}
            .transcript-box::-webkit-scrollbar-thumb {{
                background: #007bff;
                border-radius: 6px;
            }}
            .transcript-box::-webkit-scrollbar-thumb:hover {{
                background: #0056b3;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #007bff;
                color: white;
                font-weight: bold;
            }}
            .ideal-column {{
                background-color: #e3f2fd;
                font-style: italic;
                color: #0277bd;
            }}
            .emotional-table {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                overflow: hidden;
                margin: 20px 0;
            }}
            .emotional-table th {{
                background-color: #6c757d;
                color: white;
            }}
            .emotional-table td {{
                background-color: white;
            }}
            .metric-category {{
                font-weight: bold;
                background-color: #e9ecef;
                font-style: italic;
            }}
            .plot-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .plot-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .plot-item {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: #f9f9f9;
            }}
            .legend {{
                margin-top: 20px;
                padding: 15px;
                background-color: #e9ecef;
                border-radius: 5px;
            }}
            .legend-item {{
                display: inline-block;
                margin: 5px 10px;
            }}
            .legend-color {{
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-right: 5px;
                vertical-align: middle;
                border: 1px solid #000;
            }}
            .section-description {{
                background-color: #e3f2fd;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-style: italic;
                color: #0277bd;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Converser Analysis Report</h1>
            <h2>Audio File: {ID}</h2>
            
            <div class="transcript-container">
                <h3 class="transcript-header">Full Transcript</h3>
                <div class="transcript-box">{transcript_text}</div>
            </div>
            
            <h2>Core Performance Metrics</h2>
            <div class="section-description">
                These metrics measure fundamental speaking performance aspects like pace, vocal variety, and language quality.
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Score</th>
                        <th>Ideal Range</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add core metrics table rows with color coding
    for index, row in core_metrics.iterrows():
        value = row[ID]
        score = row.get('Score', None)
        color = get_score_color(score)
        ideal_value = get_ideal_value(index)
        
        # Format value based on type
        if isinstance(value, float):
            if abs(value) < 0.01:
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = f"{value:.3f}"
        else:
            formatted_value = str(value)
        
        score_text = f"{score:.0f}" if pd.notna(score) else "N/A"
        
        html_content += f"""
                    <tr style="background-color: {color};">
                        <td><strong>{index.replace('_', ' ')}</strong></td>
                        <td>{formatted_value}</td>
                        <td>{score_text}</td>
                        <td class="ideal-column">{ideal_value}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
            
            <div class="legend">
                <strong>Score Legend:</strong>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: #28a745;"></span>
                    Score 5 (Excellent)
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: #6f42c1;"></span>
                    Score 4 (Good)
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: #ffc107;"></span>
                    Score 3 (Average)
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: #fd7e14;"></span>
                    Score 2 (Below Average)
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: #dc3545;"></span>
                    Score 1 (Poor)
                </div>
            </div>
            
            <h2>Emotional & Sentiment Analysis</h2>
            <div class="section-description">
                AI-powered analysis of emotional tone, sentiment, and confidence levels throughout the speech.
            </div>
            <table class="emotional-table">
                <thead>
                    <tr>
                        <th>Analysis Type</th>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Group emotional metrics by category
    sentiment_metrics = []
    emotion_metrics = []
    confidence_metrics = []
    
    for index, row in emotional_metrics.iterrows():
        if 'Sentiment' in index:
            sentiment_metrics.append((index, row[ID]))
        elif 'Emotion' in index:
            emotion_metrics.append((index, row[ID]))
        elif 'confident' in index.lower() or 'unsure' in index.lower():
            confidence_metrics.append((index, row[ID]))
    
    # Add sentiment metrics
    if sentiment_metrics:
        html_content += '<tr class="metric-category"><td colspan="3">Sentiment Analysis (DistilBERT)</td></tr>'
        for metric, value in sentiment_metrics:
            # Clean up metric names for display
            display_name = metric.replace('DistilBERT_', '').replace('_', ' ').title()
            
            # Format value
            if isinstance(value, float):
                if 'Score' in metric or 'Percentage' in metric:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            
            html_content += f"""
                        <tr>
                            <td>Sentiment</td>
                            <td>{display_name}</td>
                            <td>{formatted_value}</td>
                        </tr>
            """
    
    # Add emotion metrics (simplified - show only key ones)
    key_emotions = ['joy', 'anger', 'fear', 'sadness', 'neutral']
    emotion_averages = [m for m in emotion_metrics if 'Average' in m[0] and any(emotion in m[0].lower() for emotion in key_emotions)]
    
    if emotion_averages:
        html_content += '<tr class="metric-category"><td colspan="3">Emotion Analysis (DistilRoBERTa)</td></tr>'
        for metric, value in emotion_averages:
            emotion_name = metric.split('_')[-1].title()
            formatted_value = f"{value:.3f}"
            html_content += f"""
                        <tr>
                            <td>Emotion</td>
                            <td>{emotion_name} Level</td>
                            <td>{formatted_value}</td>
                        </tr>
            """
        
        # Add dominant emotion
        dominant_emotion = [m for m in emotion_metrics if 'Dominant' in m[0]]
        if dominant_emotion:
            html_content += f"""
                        <tr>
                            <td>Emotion</td>
                            <td>Dominant Emotion</td>
                            <td><strong>{dominant_emotion[0][1].title()}</strong></td>
                        </tr>
            """
    
    # Add confidence metrics
    if confidence_metrics:
        html_content += '<tr class="metric-category"><td colspan="3">Confidence Analysis (BART)</td></tr>'
        for metric, value in confidence_metrics:
            if 'confident' in metric.lower():
                display_name = "Confidence Level"
            elif 'unsure' in metric.lower():
                display_name = "Uncertainty Level"
            else:
                display_name = metric.replace('BART_', '').replace('_', ' ').title()
            
            formatted_value = f"{value:.3f}"
            html_content += f"""
                        <tr>
                            <td>Confidence</td>
                            <td>{display_name}</td>
                            <td>{formatted_value}</td>
                        </tr>
            """
    
    html_content += """
                </tbody>
            </table>
            
            <h2>Metrics Across Time Windows</h2>
            <div class="plot-grid">
    """
    
    # Create plots for key metrics (exclude BART metrics from individual plotting)
    plot_metrics = [
        ('Words_Per_Minute', 'Words Per Minute', 'darkblue'),
        ('PitchStd', 'Pitch Standard Deviation (Hz)', 'blue'),
        ('LexicalDiversity', 'Lexical Diversity', 'purple'),
        ('TotalFillerCount', 'Filler Word Count', 'brown')
    ]
    
    # Define scoring functions for background colors
    def get_score_for_metric(metric, value):
        if metric == 'Words_Per_Minute':
            if 120 <= value <= 150:
                return 5
            elif 100 <= value < 120 or 150 < value <= 170:
                return 4
            elif 90 <= value < 100 or 170 < value <= 180:
                return 3
            elif 80 <= value < 90 or 180 < value <= 200:
                return 2
            else:
                return 1
        elif metric == 'PitchStd':
            if 30 <= value <= 60:
                return 5
            elif 20 <= value < 30 or 60 < value <= 70:
                return 4
            elif 15 <= value < 20 or 70 < value <= 80:
                return 3
            elif 10 <= value < 15 or 80 < value <= 90:
                return 2
            else:
                return 1
        elif metric == 'LexicalDiversity':
            if value >= 0.6:
                return 5
            elif value >= 0.5:
                return 4
            elif value >= 0.4:
                return 3
            elif value >= 0.3:
                return 2
            else:
                return 1
        elif metric == 'TotalFillerCount':
            if value == 0:
                return 5
            elif value <= 2:
                return 4
            elif value <= 5:
                return 3
            elif value <= 10:
                return 2
            else:
                return 1
        else:
            return None  # No scoring for sentiment/confidence metrics
    
    def get_background_color(score):
        if score == 5:
            return '#28a745'  # Green
        elif score == 4:
            return '#6f42c1'  # Purple
        elif score == 3:
            return '#ffc107'  # Yellow
        elif score == 2:
            return '#fd7e14'  # Orange
        elif score == 1:
            return '#dc3545'  # Red
        else:
            return '#f8f9fa'  # Light gray for no score
    
    # Plot individual metrics (excluding BART_ZeroShot metrics)
    for metric, title, color in plot_metrics:
        if metric in WindowsMetricDf.columns:
            plt.figure(figsize=(10, 6))
            
            # Calculate window centers for x-axis
            window_centers = WindowsMetricDf['WindowStart'] + (WindowsMetricDf['WindowEnd'] - WindowsMetricDf['WindowStart']) / 2
            
            # Define fixed Y-axis limits based on scoring ranges
            y_limits = {
                'Words_Per_Minute': (30, 300),      # Covers poor (below 80) to excellent (120-150) with buffer
                'PitchStd': (5, 100),               # Covers poor (below 10) to excellent (30-60) with buffer
                'LexicalDiversity': (0.1, 0.8),     # Covers poor (below 0.3) to excellent (above 0.6) with buffer
                'TotalFillerCount': (0, 15)         # Covers excellent (0) to poor (above 10) with buffer
            }
            
            # Set Y-axis limits for the current metric
            if metric in y_limits:
                plt.ylim(y_limits[metric])
            
            # Add colored background regions for each window
            for i, row in WindowsMetricDf.iterrows():
                window_start = row['WindowStart']
                window_end = row['WindowEnd']
                value = row[metric]
                
                # Get score and corresponding color
                score = get_score_for_metric(metric, value)
                if score is not None:
                    bg_color = get_background_color(score)
                    plt.axvspan(window_start, window_end, alpha=0.3, color=bg_color)
            
            # Plot the line with data points
            plt.plot(window_centers, WindowsMetricDf[metric], 
                    marker='o', linewidth=3, markersize=8, color=color, 
                    markerfacecolor='white', markeredgecolor=color, markeredgewidth=2,
                    zorder=10)  # Ensure line is on top
            
            plt.title(f'{title} Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel(title, fontsize=12)
            plt.grid(True, alpha=0.3, zorder=1)
            
            # Add legend for scoring (only for metrics that have scoring)
            if get_score_for_metric(metric, WindowsMetricDf[metric].iloc[0]) is not None:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#28a745', alpha=0.3, label='Excellent (5)'),
                    Patch(facecolor='#6f42c1', alpha=0.3, label='Good (4)'),
                    Patch(facecolor='#ffc107', alpha=0.3, label='Average (3)'),
                    Patch(facecolor='#fd7e14', alpha=0.3, label='Below Average (2)'),
                    Patch(facecolor='#dc3545', alpha=0.3, label='Poor (1)')
                ]
                plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            html_content += f"""
                <div class="plot-item">
                    <h3>{title}</h3>
                    <img src="data:image/png;base64,{plot_data}" style="max-width: 100%; height: auto;">
                </div>
            """
    
    # Create combined BART_ZeroShot confidence analysis plot
    bart_columns = [col for col in WindowsMetricDf.columns if col.startswith('BART_ZeroShot_')]
    if bart_columns:
        plt.figure(figsize=(10, 6))
        
        window_centers = WindowsMetricDf['WindowStart'] + (WindowsMetricDf['WindowEnd'] - WindowsMetricDf['WindowStart']) / 2
        
        # Define colors for each BART metric
        bart_colors = {
            'confident': 'darkgreen',
            'unsure': 'darkred',
            'neutral': 'gray'
        }
        
        for bart_col in bart_columns:
            metric_name = bart_col.replace('BART_ZeroShot_', '')
            color = bart_colors.get(metric_name, 'black')
            
            plt.plot(window_centers, WindowsMetricDf[bart_col], 
                    marker='o', linewidth=2, markersize=6, 
                    label=metric_name.title(), color=color, 
                    markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
        
        plt.title('Confidence Analysis Over Time (BART)', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)  # Set y-axis limits for confidence scores
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        html_content += f"""
            <div class="plot-item">
                <h3>Confidence Analysis Over Time (BART)</h3>
                <img src="data:image/png;base64,{plot_data}" style="max-width: 100%; height: auto;">
            </div>
        """
    
    # Add emotion scores plot if available
    emotion_columns = [col for col in WindowsMetricDf.columns if col.startswith('DistilRoBERTa_Emotion_')]
    if emotion_columns:
        plt.figure(figsize=(10, 6))
        
        window_centers = WindowsMetricDf['WindowStart'] + (WindowsMetricDf['WindowEnd'] - WindowsMetricDf['WindowStart']) / 2
        
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
        for i, emotion_col in enumerate(emotion_columns):
            emotion_name = emotion_col.replace('DistilRoBERTa_Emotion_', '').capitalize()
            color = colors[i % len(colors)]
            plt.plot(window_centers, WindowsMetricDf[emotion_col], 
                    marker='o', linewidth=2, markersize=4, label=emotion_name, color=color)
        
        plt.title('Emotion Scores Over Time (DistilRoBERTa)', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Emotion Score', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        html_content += f"""
            <div class="plot-item">
                <h3>Emotion Analysis Over Time (DistilRoBERTa)</h3>
                <img src="data:image/png;base64,{plot_data}" style="max-width: 100%; height: auto;">
            </div>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    html_file_path = os.path.join(output_dir, f"{ID}_Report.html")
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report created: {html_file_path}")




###############################################################################
###############################################################################
###############################################################################


# Set up argument Parser
parser = argparse.ArgumentParser(description="Analyze public speaking audio files.")
parser.add_argument("--input_file", type=str, help="Path to the input .m4a file.", required=True)
parser.add_argument("--time_window", type=int, default=60, help="Time window size in seconds (default: 60).")
parser.add_argument("--output_dir",type=str,help="Path to the output directory",default='./')
parser.add_argument("--user_label",type=str,default='')
# Parse the arguments
args = parser.parse_args()
# Extract ID
ID = args.input_file.split('/')[-1].replace('.m4a', '')

# Create output dir  if it doesnt exist
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)


# 1) Convert to wav file 
args.output_wav = args.input_file.replace('.m4a', '.wav')
convert_m4a_to_wav(args.input_file, args.output_wav)


# 2) Get the length of the wav file
Duration = get_wav_length(args.output_wav)

# 2) Run Whisper and get the Transcript. 
WhisperResults = WhisperCreateTranscript(args.output_wav, Model='large')
Transcript = WhisperResults['text']

# 3) Combine all segments together.
WordTimestamps = []
for a in WhisperResults['segments']:
    WordTimestamps += a['words']


WindowMetricList = []
EmotionScore = defaultdict(float)
SegmentCount=0

AllPitch = pd.DataFrame()
AllPauses = pd.DataFrame()

WindowStart = 0
WindowEnd = args.time_window


while True:  # Infinite loop, explicitly break when no words are left
    WindowMetrics = {}
    WindowMetrics['WindowStart'] = WindowStart
    WindowMetrics['WindowEnd'] = WindowEnd

    # Subset the word list based on the time window
    WordsInWindow = [word for word in WordTimestamps if WindowStart <= word['start'] < WindowEnd]

    # Subset of the transcript in use
    TempTranscript = ' '.join([x['word'] for x in WordsInWindow])

     # Break the loop if no words are found in the current window
    if not WordsInWindow: 
        break

    # Calculate word rate in the current time window
    WordRate = len(WordsInWindow)
    WindowMetrics['WordRate'] = WordRate

    # Calculate actual window duration (may be shorter for the last window)
    actual_window_duration = min(args.time_window, Duration - WindowStart)
    WindowMetrics['ActualWindowDuration'] = actual_window_duration

    # Calculate words per minute for this specific window
    if actual_window_duration > 0:
        WindowMetrics['Words_Per_Minute'] = (WordRate / actual_window_duration) * 60
    else:
        WindowMetrics['Words_Per_Minute'] = 0

    # Extract pitch data for the words in the current time window
    WordPitchData, WindowMetrics['PitchStd'], WindowMetrics['PitchMean'] = extract_word_pitch_data(args.output_wav, WordsInWindow)

    # Extract pauses and gaps between words
    WindowMetrics['MeanTimeGap'], TimeGaps, WindowMetrics['Pauses'] = extract_pauses(WordsInWindow)

    # Calculate lexical diversity for the current time window
    WindowMetrics['LexicalDiversity'] = lexical_diversity(' '.join([word['word'] for word in WordsInWindow]))

    # Detect filler words in the current time window
    FillerWordCounts, WindowMetrics['TotalFillerCount'] = detect_filler_words(' '.join([word['word'] for word in WordsInWindow]))

    # Window-level Sentiment Analysis
    if TempTranscript.strip():  # Only analyze if there's actual text
        window_sentiment = analyze_sentiment_transformers(TempTranscript)
        WindowMetrics['DistilBERT_Sentiment_Label'] = window_sentiment[0]['label']
        WindowMetrics['DistilBERT_Sentiment_Score'] = window_sentiment[0]['score']
        
        # Window-level General Emotion Analysis
        window_emotion = analyze_emotions(TempTranscript)
        for e in window_emotion[0]:
            WindowMetrics[f'DistilRoBERTa_Emotion_{e["label"]}'] = e['score']
        highest_emotion = max(window_emotion[0], key=lambda x: x['score'])
        WindowMetrics['DistilRoBERTa_Dominant_Emotion'] = highest_emotion['label']
        
        # Window-level Zero Shot Emotional Analysis
        zero_shot_emotion_analysis = Zero_Shot_Analyse_Emotions(TempTranscript)
        for zse in zero_shot_emotion_analysis:
            WindowMetrics[f'BART_ZeroShot_{zse[0]}'] = zse[1]
            EmotionScore[zse[0]] += zse[1]
        
        # Store dominant zero-shot emotion for this window
        WindowMetrics['BART_Dominant_ZeroShot_Emotion'] = zero_shot_emotion_analysis[0][0]
    else:
        # Handle empty windows
        WindowMetrics['DistilBERT_Sentiment_Label'] = 'NEUTRAL'
        WindowMetrics['DistilBERT_Sentiment_Score'] = 0.5
        WindowMetrics['DistilRoBERTa_Dominant_Emotion'] = 'neutral'
        WindowMetrics['BART_Dominant_ZeroShot_Emotion'] = 'neutral'

    # Merge results
    if WindowMetrics:
        WindowMetricList.append(WindowMetrics)
        AllPitch = pd.concat([AllPitch, WordPitchData], ignore_index=True)
        AllPauses = pd.concat([AllPauses, TimeGaps], ignore_index=True)

    # Update Window Metrics
    WindowStart = WindowEnd
    WindowEnd += args.time_window
    SegmentCount += 1


# Calculate Summary Metrics
WindowsMetricDf = pd.DataFrame(WindowMetricList)
AllPitch = AllPitch.reset_index(drop=True)
AllPauses = AllPauses.reset_index(drop=True)

# Convert Emotional Score to DataFrame And Normalise by Segments
EmotionalScoreDf = pd.DataFrame([EmotionScore])
EmotionalScoreDf = EmotionalScoreDf.T
EmotionalScoreDf[0] = EmotionalScoreDf[0]/SegmentCount 


SummaryDict = {}
SummaryDict['ID'] = ID
SummaryDict['Date'] = pd.to_datetime('today').strftime('%Y-%m-%d')
SummaryDict['User_Label'] = args.user_label
SummaryDict['Duration'] = Duration
SummaryDict['Word Count'] = len(Transcript.split(' '))
SummaryDict['Average_Words_Per_Minute'] = len(Transcript.split(' ')) / (Duration / 60)
SummaryDict['Mean_Pitch'] = AllPitch['mean_pitch'].mean()
SummaryDict['Mean_Pitch_Std'] = AllPitch['mean_pitch'].std()
SummaryDict['Mean_Pause'] = AllPauses['gap'].mean()
SummaryDict['Mean_Pause_Std'] = AllPauses['gap'].std()
SummaryDict['Total_Filler_Count'] = WindowsMetricDf['TotalFillerCount'].sum()
SummaryDict['Overall_Lexical_Diversity'] = lexical_diversity(Transcript)
SummaryDict['Overall_Readability'] = readability_grade(Transcript)

# Aggregate sentiment analysis from windows
positive_windows = sum(1 for w in WindowMetricList if w.get('DistilBERT_Sentiment_Label') == 'POSITIVE')
negative_windows = sum(1 for w in WindowMetricList if w.get('DistilBERT_Sentiment_Label') == 'NEGATIVE')
neutral_windows = sum(1 for w in WindowMetricList if w.get('DistilBERT_Sentiment_Label') == 'NEUTRAL')

SummaryDict['DistilBERT_Positive_Windows_Count'] = positive_windows
SummaryDict['DistilBERT_Negative_Windows_Count'] = negative_windows
SummaryDict['DistilBERT_Neutral_Windows_Count'] = neutral_windows
SummaryDict['DistilBERT_Positive_Windows_Percentage'] = (positive_windows / len(WindowMetricList)) * 100
SummaryDict['DistilBERT_Negative_Windows_Percentage'] = (negative_windows / len(WindowMetricList)) * 100
SummaryDict['DistilBERT_Neutral_Windows_Percentage'] = (neutral_windows / len(WindowMetricList)) * 100

# Overall dominant sentiment (most frequent across windows)
if positive_windows >= negative_windows and positive_windows >= neutral_windows:
    SummaryDict['DistilBERT_Overall_Dominant_Sentiment'] = 'POSITIVE'
elif negative_windows >= neutral_windows:
    SummaryDict['DistilBERT_Overall_Dominant_Sentiment'] = 'NEGATIVE'
else:
    SummaryDict['DistilBERT_Overall_Dominant_Sentiment'] = 'NEUTRAL'

# Average sentiment scores across windows
sentiment_scores = [w.get('DistilBERT_Sentiment_Score', 0.5) for w in WindowMetricList if 'DistilBERT_Sentiment_Score' in w]
SummaryDict['DistilBERT_Average_Sentiment_Score'] = np.mean(sentiment_scores) if sentiment_scores else 0.5

# Aggregate general emotions from windows
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']  # Adjust based on your model
for emotion in emotion_labels:
    emotion_scores = [w.get(f'DistilRoBERTa_Emotion_{emotion}', 0) for w in WindowMetricList]
    SummaryDict[f'DistilRoBERTa_Average_Emotion_{emotion}'] = np.mean(emotion_scores) if emotion_scores else 0

# Most frequent dominant general emotion across windows
general_emotions = [w.get('DistilRoBERTa_Dominant_Emotion', 'neutral') for w in WindowMetricList]
from collections import Counter

emotion_counts = Counter(general_emotions)
SummaryDict['DistilRoBERTa_Overall_Dominant_Emotion'] = emotion_counts.most_common(1)[0][0]

# Aggregate zero-shot emotions (already accumulated in EmotionScore)
SummaryDict['BART_Dominant_ZeroShot_Emotion'] = EmotionalScoreDf.sort_values(by=[0], ascending=False).index.tolist()[0]

# Zero-shot emotion averages across windows
for emotion in ['confident', 'unsure', 'neutral']:  # Adjust based on your candidate labels
    SummaryDict[f'BART_Average_ZeroShot_{emotion}'] = EmotionScore[emotion] / SegmentCount


# Convert to DataFrame
SummaryDf = pd.DataFrame(SummaryDict, index=[ID]).T

# Score Metrics which can be scored.
SummaryDf = score_metrics(SummaryDf,ID)


##########################
#  Write Output Files    #
##########################

# Create the transcript file path
TranscriptFile = open(os.path.join(args.output_dir, f"{ID}_Transcript.txt"), 'w')
TranscriptFile.write(Transcript)
TranscriptFile.close()
SummaryDf.to_csv(os.path.join(args.output_dir, f"{ID}_Summary.csv"), index=True)
WindowsMetricDf.to_csv(os.path.join(args.output_dir, f"{ID}_WindowMetrics.csv"), index=False)
AllPitch.to_csv(os.path.join(args.output_dir, f"{ID}_Pitch.csv"), index=False)
AllPauses.to_csv(os.path.join(args.output_dir, f"{ID}_Pauses.csv"), index=False)
EmotionalScoreDf.to_csv(os.path.join(args.output_dir, f"{ID}_Emotion.csv"), index=False)

# Create HTML report
create_html_report(SummaryDf, WindowsMetricDf, ID, args.output_dir)


# Automatically open the HTML report in the default browser
html_file_path = os.path.join(args.output_dir, f"{ID}_Report.html")
webbrowser.open(f'file://{os.path.abspath(html_file_path)}')
print(f"Opening HTML report in browser: {html_file_path}")
