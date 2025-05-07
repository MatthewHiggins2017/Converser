## Converser 

**Description**: Python package to assess and improve public speaking. 

## Installation.

Use the commands below to install the package. 

```
git clone https://github.com/MatthewHiggins2017/Converser.git
cd Converser
conda env create -f Converser.yaml
conda activate Converser
pip install . 

```

## Inputs

```
# Example Command
Converser_Assess.py --input_file ./Path/To/Audio/File 
```


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
| **Mean Pause Length (sec)**         | 0.1–0.3                                | 0.05–0.09 or 0.31–0.5 | <0.05 or 0.51–0.7 | 0.71–1.0         | >1.0               |
| **Filler Word Count**               | 0                                      | 1–2                   | 3–5               | 6–10             | >10                |
| **Lexical Diversity (TTR)**         | ≥ 0.6                                  | 0.5–0.59              | 0.4–0.49          | 0.3–0.39         | <0.3               |
| **Readability Grade**               | 7–9 (ideal range for general audience) | 6 or 10               | 5 or 11           | 4 or 12          | ≤3 or ≥13          |
