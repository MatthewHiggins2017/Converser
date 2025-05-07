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

# import nltk
# nltk.download('punkt_tab', quiet=True)

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
        "AverageWordsPerMinute": score_wpm,
        "Mean_Pitch_Std": score_pitch_std,
        "Mean_Pause": score_pause,
        "TotalFillerCount": score_fillers,
        "OverallLexicalDiversity": score_lexical_diversity,
        "Overall Readability": score_readability,
    }

    # Initialize the "Score" column with NaN
    df["Score"] = np.nan

    # Calculate scores for each metric if applicable
    for metric, scoring_function in metrics_to_score.items():
        if metric in df.index:
            df.at[metric, "Score"] = scoring_function(df.loc[metric, ID])

    return df


###############################################################################
###############################################################################
###############################################################################


# Set up argument parser
parser = argparse.ArgumentParser(description="Analyze public speaking audio files.")
parser.add_argument("--input_file", type=str, help="Path to the input .m4a file.", required=True)
parser.add_argument("--time_window", type=int, default=30, help="Time window size in seconds (default: 30).")
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

    # Subset of the transcript in use
    TempTranscript = ' '.join([x['word'] for x in WordsInWindow])

     # Break the loop if no words are found in the current window
    if not WordsInWindow: 
        break

    # Calculate word rate in the current time window
    WordRate = len(WordsInWindow)

    # Extract pitch data for the words in the current time window
    WordPitchData, WindowMetrics['PitchStd'], WindowMetrics['PitchMean'] = extract_word_pitch_data(args.output_wav, WordsInWindow)

    # Extract pauses and gaps between words
    WindowMetrics['MeanTimeGap'], TimeGaps, WindowMetrics['Pauses'] = extract_pauses(WordsInWindow)

    # Calculate lexical diversity for the current time window
    WindowMetrics['LexicalDiversity'] = lexical_diversity(' '.join([word['word'] for word in WordsInWindow]))

    # Detect filler words in the current time window
    FillerWordCounts, WindowMetrics['TotalFillerCount'] = detect_filler_words(' '.join([word['word'] for word in WordsInWindow]))

    # Run Emotional Analysis
    zero_shot_emotion_analysis = Zero_Shot_Analyse_Emotions(Transcript)
    for zse in zero_shot_emotion_analysis:
        EmotionScore[zse[0]]+=zse[1]

    # Merge results
    if WindowMetrics:
        WindowMetricList.append(WindowMetrics)
        AllPitch = pd.concat([AllPitch, WordPitchData], ignore_index=True)
        AllPauses = pd.concat([AllPauses, TimeGaps], ignore_index=True)

    # Update Window Metrics
    WindowStart = WindowEnd
    WindowEnd += args.time_window
    SegmentCount +=1


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

# Full Transcript Sentiment Analysis (Positive /Negative /Neutral)
sentiment_result = analyze_sentiment_transformers(Transcript)
SummaryDict['Sentiment_Label'] = sentiment_result[0]['label']  
SummaryDict['Sentiment_Score'] = sentiment_result[0]['score']

# Full Transcript General Emotion Analysis 
emotion_result = analyze_emotions(Transcript)
for e in emotion_result[0]:
    SummaryDict[f'General_Emotion_{e["label"]}'] = e['score']
highest_emotion = max(emotion_result[0], key=lambda x: x['score'])
SummaryDict['Dominant_General_Emotion'] = highest_emotion['label']

# Zero Shot Emotion Classification 
SummaryDict['Dominant_Zero_Shot_Emotion'] = EmotionalScoreDf.sort_values(by=[0],ascending=False).index.tolist()[0]

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

print('\n\n')
print(Transcript)
print('\n\n')
print(SummaryDf)
print('\n\n')
print(EmotionalScoreDf)


###############################################################################
###############################################################################
