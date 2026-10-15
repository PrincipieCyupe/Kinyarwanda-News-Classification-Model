# Kinyarwanda News Classifier

A text classification system that assigns a Kinyarwanda news article to one of 14 categories (politics, sport, economy, health, entertainment, history, technology, tourism, culture, fashion, religion, environment, education, relationship).

## Project links

- Live app: [https://kinnews.streamlit.app](https://kinnews.streamlit.app)
- Demo video: [Demo Video - MLT1 Summative Project](https://drive.google.com/file/d/1wmhAvltHoafeLhDk29wgOiOl5V0xkC6S/view?usp=sharing)
- Dataset release (KINNEWS, Niyongabo et al. 2020): [https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus)

## Problem

Rwandan news websites such as igihe.com publish articles under category sections, and today each article is tagged by hand by an editor before publishing. A classifier that reads an article and returns its predicted category removes that manual step. The task is multi-class classification across 14 Kinyarwanda news categories.

## Dataset

**Source.** KINNEWS, released by Niyongabo, Qu, Kreutzer and Huang (2020) through the GitHub repository linked above. 21,268 Kinyarwanda news articles collected from 15 Rwandan websites and 5 newspapers, split by the authors into 17,014 training and 4,254 test articles. The raw version was used so preprocessing could be controlled and documented end to end.

**Preparation.** Five test rows with missing `content` were dropped (0.12 percent of test), leaving 4,249 test articles. A stratified 10 percent slice of the training set was carved out as validation, leaving 15,312 training and 1,702 validation articles. The authors' test set was never touched during training or model selection.

**Preprocessing.** For every article the pipeline concatenates title and content, lowercases, strips URLs and non-alphabetic characters, removes the 80-word Kinyarwanda stopword set released with KINNEWS, drops tokens shorter than two characters, and drops rows that become empty after cleaning. The function is pure and deterministic and is reused verbatim at inference time inside the Streamlit app.

**Challenges.** The training set is heavily imbalanced (politics 26 percent, environment 0.22 percent). The authors' test set has a very different class distribution from training (politics drops to 10 percent, economy from 25 to 5 percent, several small classes grow by 10x to 20x), which caps the test accuracy achievable with any model trained on this split. This is quantified in the report.

## Methodology

The pipeline is: raw text → preprocessing → tokenisation → model → class.

Two families of models were trained under identical experimental conditions and compared on the authors' held-out test split.

**Classical.** TF-IDF representation (up to 50,000 features, unigrams and bigrams, min document frequency 2) feeding either Logistic Regression or Linear SVM. The first classical classifier (Logistic Regression) was trained twice: with default class weights and with balanced class weights, so the effect of imbalance correction could be measured. Where class balance was a better approach and used throughout the rest experiments

**Deep sequential.** Padded integer sequences of length 650 (chosen to cover the 95th percentile of article length) with a vocabulary cap of 80,000 tokens (covers 98 percent of training token occurrences). Three variants of a Bidirectional GRU (64 hidden units) differing only in the embedding matrix: a 50-dimensional Word2Vec trained from scratch on the training texts (gensim, skip-gram, 10 epochs), the authors' pretrained W2V-Kin-50, and the authors' pretrained W2V-Kin-100. A fourth deep model replaced the BiGRU with a 1D convolutional block (128 filters, kernel size 5, global max pooling) on top of W2V-Kin-100.

Every deep model used SpatialDropout1D (0.3) after the embedding, Dropout (0.4) before the final dense softmax over 14 classes, Adam optimiser, sparse categorical cross-entropy, EarlyStopping on validation loss with patience 4, batch size 64, up to 20 epochs. Balanced class weights computed with scikit-learn's `compute_class_weight` were passed to every deep model so minority classes contributed proportionally to the loss. Pretrained embeddings were allowed to fine-tune during training.

## Model

The deployed model is **TF-IDF + Logistic Regression with balanced class weights (E2)**. It won on both accuracy and macro F1 across all seven experiments. Why this won:

- TF-IDF ignores the exact order of tokens and relies on which words appear and how often, which is less sensitive to the train-test distribution shift than embeddings that have been fine-tuned on the training distribution.
- Balanced class weights pushed Logistic Regression's macro F1 from 0.415 to 0.575, a 16-point gain, by making the model care equally about minority classes.
- On this corpus, deep sequential models reached validation accuracy above 85 percent but dropped roughly 30 points on test. That gap is a distribution-shift problem, not a modelling bug.

## Experiments and results

| Rank | Experiment | Accuracy | Macro P | Macro R | Macro F1 |
|---|---|---|---|---|---|
| 1 | E2 TF-IDF + Logistic Regression (balanced) | 0.7110 | 0.6478 | 0.6249 | **0.5754** |
| 2 | E3 TF-IDF + Linear SVM (balanced) | 0.7119 | 0.6395 | 0.5737 | 0.5415 |
| 3 | E1 TF-IDF + Logistic Regression | 0.6540 | 0.5424 | 0.4596 | 0.4151 |
| 4 | E6 BiGRU + Pretrained W2V-Kin-100 | 0.5670 | 0.4856 | 0.4311 | 0.4086 |
| 5 | E7 1D CNN + Pretrained W2V-Kin-100 | 0.4361 | 0.5120 | 0.4657 | 0.4069 |
| 6 | E5 BiGRU + Pretrained W2V-Kin-50 | 0.5291 | 0.4546 | 0.4501 | 0.3971 |
| 7 | E4 BiGRU + Own-trained W2V-50 | 0.4912 | 0.4741 | 0.4458 | 0.3870 |

**Best per-class F1 (E2):** sport 0.94, religion 0.81, entertainment 0.80, relationship 0.78, technology 0.68, tourism 0.67, health 0.64, politics 0.62, fashion 0.62, economy 0.55, environment 0.43, culture 0.28, education 0.14, history 0.09.

**Honest reading.** The paper reports 88.65 percent accuracy on this test set with BiGRU plus W2V-Kin-50. The 17-point gap to our 71.10 is explained by two measured factors: the class distribution mismatch between train and test (validated by running the same LogReg on validation and test, which gave 87.78 vs 65.40 accuracy), and the use of the raw version with our own preprocessing rather than the authors' cleaned version. Both are documented in the report.

## Reproduction

### Train and evaluate (notebook) on Google Colab

The notebook is designed to run end to end on a free Colab GPU runtime. **Highly recommended path, not local.**

1. **Download the dataset.** Open the KINNEWS release page (https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus) and download the files for Kinyarwanda below from the Drive links in the KINNEWS README:
   - Raw KINNEWS CSVs (gives you `train.csv` and `test.csv` inside a `raw/` folder).
   - Pretrained Kinyarwanda Word2Vec, 50-dim and 100-dim (`W2V-Kin-50.txt` and `W2V-Kin-100.txt`).
   - `classes.txt` and `stopwords/Kinyarwanda/Kinyarwanda stopwords set.txt` from the GitHub repo itself (rename the stopwords file to `stopwords.txt`).

2. **Set up your Google Drive** to look exactly like this (the paths are referenced from the notebook):
   ```
   MyDrive/KINNEWS/
   ├── raw/
   │   ├── train.csv
   │   └── test.csv
   ├── embeddings/
   │   ├── W2V-Kin-50.txt
   │   └── W2V-Kin-100.txt
   ├── classes.txt
   └── stopwords.txt
   ```

3. **Open the notebook in Colab.** Upload `notebook/PrincipieCyubahiro_Summative_Assignment_MLT1.ipynb` to Colab, or open it from GitHub by prefixing the file's GitHub URL with `https://colab.research.google.com/github/` as Colab's "Open on GitHub" picker does.

4. **Set the runtime to GPU** (Runtime → Change runtime type → GPU). The deep models train in minutes; CPU-only is noticeably slower.

5. **Run top to bottom.** The notebook will:
   - mount your Drive
   - load and clean the raw CSVs
   - run the exploratory analysis
   - build the train/validation/test splits and the tokenizer
   - load the pretrained embeddings and train a Word2Vec on the training texts
   - run all seven experiments and save per-experiment metrics and reports
   - do the error analysis
   - write the deployment bundle to `MyDrive/KINNEWS/outputs/deploy/`

   The whole run takes roughly 15 to 30 minutes on a free Colab T4.

6. **Copy the deployment bundle into this repo.** After the notebook finishes, download the five files from `MyDrive/KINNEWS/outputs/deploy/` and place them in this repo's `deploy/` folder (already present from the committed version). This is the only artefact the Streamlit app needs.

### Run the Streamlit app

The app is a single page that loads the deployed classifier and classifies any Kinyarwanda article the user pastes in.

Local:

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# or on Windows:  venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

Deploy on Streamlit Community Cloud:

1. Push this repository to GitHub (public).
2. On share.streamlit.io, click **New app**, point it at this repo, pick `main` as the branch, pick `app.py` as the main file, click **Deploy**.
3. First deploy takes 1 to 3 minutes while Streamlit installs the dependencies in `requirements.txt`.

### Fallback if the live app is offline at the time of grading

Streamlit Community Cloud puts inactive apps to sleep. If the live link does not respond, the grader can run the app locally in under a minute using the steps above (the `deploy/` folder is already in this repo, so no notebook rerun is needed).

## Repository layout

```
.
├── app.py                   Streamlit app (single-page predict interface)
├── .streamlit/config.toml   Streamlit theme and server config
├── requirements.txt         Pinned runtime dependencies
├── deploy/                  The deployed model bundle (shipped with the repo)
│   ├── model.pkl            TF-IDF + Logistic Regression (balanced)
│   ├── vectorizer.pkl       Fitted TF-IDF vectoriser
│   ├── label_map.json       Index to class-name mapping (0 to 13)
│   ├── config.json          Model metadata read by the app
│   └── stopwords.txt        Kinyarwanda stopword list used in preprocessing
├── notebook/                End-to-end training and evaluation notebook
└── report/                  Final summative report (PDF)
```

## Acknowledgements

- Niyongabo, R. A., Qu, H., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. COLING 2020. Dataset, stopword list and pretrained W2V-Kin embeddings used in this project.
- Libraries: scikit-learn, numpy, gensim, TensorFlow Keras (notebook), Streamlit (app).

## Author

Principie Cyubahiro, BSc Software Engineering, African Leadership University, Kigali. p.cyubahiro@alustudent.com
