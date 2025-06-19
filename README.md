![Converser Banner](./Icon.png)

**Description**: Python package to assess and improve public speaking. 

## Installation

Use the commands below to install the package. 

```
git clone https://github.com/MatthewHiggins2017/Converser.git
cd Converser
conda env create -f Converser.yaml
conda activate CONVERSER
pip install . 
```


## Usage

Run the analysis script with the following command. The only required argument is the input audio file.

```bash
python scripts/Converser_Assess.py -i <path_to_audio_file> [options]
```

**Example:**
```bash
python scripts/Converser_Assess.py -i ./Test/Test_1.m4a -o ./Test/Test_1_Out
```

### Command-Line Arguments

*   `-i`, `--input_file`: (Required) Path to the input audio file (e.g., `.m4a`, `.mp3`).
*   `-o`, `--output_dir`: (Optional) Path to the output directory. Defaults to the current directory (`./`).
*   `-t`, `--time_window`: (Optional) Time window size in seconds for analysis. Defaults to `60`.
*   `--Zero_Shot_Labels`: (Optional) Comma-separated list of custom labels for zero-shot emotion analysis. Defaults to `confident,unsure`.

## Output

The script generates a comprehensive set of output files in the specified output directory, including:

*   **HTML Report (`<ID>_Report.html`)**: A detailed, interactive report that opens automatically in your web browser. It includes:
    *   The full speech transcript.
    *   A summary table of core performance metrics with scores.
    *   Emotional and sentiment analysis results.
    *   Plots showing how metrics like speech rate, pitch, and confidence change over time.
*   **Transcript (`<ID>_Transcript.txt`)**: A plain text file containing the speech transcript.
*   **Summary CSV (`<ID>_Summary.csv`)**: A CSV file with all the final summary metrics and scores.
*   **Detailed Data CSVs**: Several CSV files for in-depth analysis:
    *   `_WindowMetrics.csv`: Metrics calculated for each time window.
    *   `_Pitch.csv`: Word-level pitch data.
    *   `_Pauses.csv`: Data on gaps between words.
    *   `_Emotion.csv`: Aggregated emotion scores.


## Goals
* Aim for speech rate that feels natural (120–150 WPM).
* Pitch variation should show expressiveness without being erratic.
* Pauses should be used for clarity and emphasis, not filler.
* Zero filler words is best, but 1–2 is acceptable.
* High lexical diversity signals articulate and varied vocabulary.
* Readability should match the audience (Grade 8–9 for public, 12–13 for academic).

| **Metric**                          | **Score 5 (Ideal)**                    | **Score 4**           | **Score 3**       | **Score 2**      | **Score 1 (Poor)** |
| ----------------------------------- | -------------------------------------- | --------------------- | ----------------- | ---------------- | ------------------ |
| **Average Words Per Minute (WPM)**  | 120–150                                | 100–119 or 151–170    | 90–99 or 171–180  | 80–89 or 181–200 | <80 or >200        |
| **Mean Pitch Variability (Hz std)** | 30–60                                  | 20–29 or 61–70        | 15–19 or 71–80    | 10–14 or 81–90   | <10 or >90         |
| **Mean Pause Duration (s)**         | 0.1–0.3                                | 0.05–0.1 or 0.3–0.5   | <0.05 or 0.5–0.7  | 0.7–1.0          | >1.0               |
| **Filler Word Count**               | 0                                      | 1–2                   | 3–5               | 6–10             | >10                |
| **Lexical Diversity (TTR)**         | ≥ 0.6                                  | 0.5–0.59              | 0.4–0.49          | 0.3–0.39         | <0.3               |
| **Readability Grade**               | 7–9 (ideal range for general audience) | 6 or 10               | 5 or 11           | 4 or 12          | ≤3 or ≥13          |




## Example Audio Sources

The analysis examples (see Examples folder) for the speeches by Winston Churchill and Martin Luther King Jr. were generated using the following audio files:

*   **Martin Luther King Jr. ("I Have a Dream")**: [`https://ia801605.us.archive.org/25/items/MLKDream/MLKDream_64kb.mp3`](https://ia801605.us.archive.org/25/items/MLKDream/MLKDream_64kb.mp3)
*   **Winston Churchill ("We Shall Fight on the Beaches")**: [`https://s3.amazonaws.com/RE-Warehouse/w/winston_churchill_speeches_and_radio_broadcasts_1940-06-04_we_shall_never_surrender.mp3`](https://s3.amazonaws.com/RE-Warehouse/w/winston_churchill_speeches_and_radio_broadcasts_1940-06-04_we_shall_never_surrender.mp3)



## Transformer-Based Models Used

This package leverages several state-of-the-art transformer models to analyze speech and text data:

### 1. **Whisper Model**
   - **Purpose**: Converts audio to text for transcription.
   - **Model**: `whisper` (by OpenAI)
   - **Description**: Whisper is a robust automatic speech recognition (ASR) model capable of handling multilingual and multitask transcription. It is designed to work well with diverse accents, background noise, and technical jargon.

### 2. **DistilBERT for Sentiment Analysis**
   - **Purpose**: Analyzes the sentiment of the transcript (e.g., positive, negative, or neutral).
   - **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
   - **Description**: A lightweight version of BERT fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset. It is optimized for sentiment classification tasks while being faster and more efficient than the original BERT.
   - **Function**: Encoded by the analyze_sentiment_transformers() function.

### 3. **DistilRoBERTa for Emotion Analysis**
   - **Purpose**: Detects emotions in the transcript, such as joy, sadness, anger, etc.
   - **Model**: `j-hartmann/emotion-english-distilroberta-base`
   - **Description**: A fine-tuned version of DistilRoBERTa trained on emotion datasets like GoEmotions. It provides detailed emotion scores for nuanced analysis.
   - **Function**: Encoded by the analyze_emotions() function.

### 4. **BART for Zero-Shot Classification**
   - **Purpose**: Performs zero-shot emotion classification, identifying emotions without requiring task-specific training.
   - **Model**: `facebook/bart-large-mnli`
   - **Description**: BART is a sequence-to-sequence model fine-tuned for natural language inference (NLI). It enables zero-shot classification by comparing input text against predefined labels, making it highly flexible for emotion detection.
   - **Function**: Encoded by the Zero_Shot_Analyse_Emotions() function.
   - **Note**: The `facebook/bart-large-mnli` model is highly effective, but newer models like `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` may offer improved performance and are being considered for future updates.



---------------

## Model Summmary

| Feature        | DistilBERT (Sentiment)                            | DistilRoBERTa (Emotion)                         | BART (Zero-Shot)                             |
| -------------- | ------------------------------------------------- | ----------------------------------------------- | -------------------------------------------- |
| Model          | `distilbert-base-uncased-finetuned-sst-2-english` | `j-hartmann/emotion-english-distilroberta-base` | `facebook/bart-large-mnli`                   |
| Task           | Sentiment (binary)                                | Emotion (multi-label)                           | Zero-shot classification                     |
| Training Data  | SST-2                                             | GoEmotions (and similar)                        | MNLI (entailment)                            |
| Flexibility    | Low                                               | Medium                                          | High                                         |
| Output         | Positive / Negative                               | Joy, Sadness, Anger, etc.                       | Custom-defined labels                        |
| Speed          | Fast                                              | Fast                                            | Slower                                       |
| Ideal Use Case | Quick polarity sentiment check                    | Rich emotional analysis                         | Dynamic emotion detection without retraining |



These models collectively enable Converser to provide comprehensive insights into speech patterns, sentiment, and emotional tone.

------------


## Scoring System In More Detail



### Readability (Flesch-Kincaid Grade Level)

The Flesch-Kincaid Grade Level is a readability test estimating the years of education required to understand a text. A lower score indicates easier readability.

**Algorithm Formula:**
$$0.39 \times (\text{words} / \text{sentences}) + 11.8 \times (\text{syllables} / \text{words}) - 15.59$$


#### Interpreting Specific Grade Levels

* **0-4 (Beginning to 4th Grade):** Very simple texts, often for young children or extremely basic instructions where clarity is paramount.
* **5th-6th Grade:** Very easy to read, with shorter sentences and simpler words. Easily understood by average 11-year-olds.
* **7th-9th Grade:** Fairly easy to read, considered "plain English." This is the sweet spot for content aimed at a broad adult audience.
* **10th-12th Grade (High School):** Fairly difficult to read. Appropriate for audiences with higher education or complex topics requiring specialized vocabulary.
* **13+ (College Level and above):** Difficult to very difficult to read. Typically reserved for academic papers, highly specialized technical manuals, or legal documents where precision and comprehensive detail are prioritized over broad accessibility.


**Ideal Score for General Population:**
Aim for a score between **7.0 and 9.0** to ensure your text is easily understood by a broad audience.

##### More to Come!

------------