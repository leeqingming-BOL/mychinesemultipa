# Pulse: Chinese Speech-to-IPA with High-Quality Annotation

![Pulse: Chinese Speech2IPA with High-Quality Annotation](poster/poster.jpg)

## Introduction

Pulse is a specialized model for converting Chinese speech to International Phonetic Alphabet (IPA) transcriptions. It addresses the unique challenges of Chinese as a tonal language by integrating semi-automatic annotations with a tone-optimized architecture. Unlike generic models, Pulse is specifically designed to handle Chinese tonal complexities and phonological characteristics.

This project has both clinical and technical significance. From a clinical perspective, it provides a tool for early identification of language developmental issues in children, supporting timely intervention for conditions like Developmental Language Disorder (DLD). From a technical perspective, it advances the state-of-the-art in speech-to-IPA conversion for tonal languages.

## Features and Innovations

1. **High-Quality Chinese IPA Annotation Dataset**: We developed a refined annotation approach inspired by [Taguchi et al. (2023)](https://arxiv.org/abs/2308.03917), ensuring accurate tone and phoneme representation.

2. **Joint Phoneme-Tone Modeling**: Our architecture is tailored to Chinese characteristics, explicitly modeling interactions between segmental phonemes and suprasegmental tones through self-attention mechanisms.

3. **Enhanced Evaluation Methods**: We implement multiple evaluation metrics including:
   - Character Error Rate (CER)
   - Phone Error Rate (PER)
   - Phone Feature Error Rate (PFER)
   - Levenshtein Distance
   - Feature-weighted Phone Error Rate

4. **Data Efficiency**: Our model achieves superior performance with just 1,000 training samples (16.2% PFER), outperforming both Allosaurus (20.9% PFER) and Wav2Vec2Phoneme (18.3% PFER).

5. **Direct End-to-End Conversion**: We achieve direct speech-to-IPA transcription without requiring intermediate grapheme representation.

## Installation

```bash
# Clone the repository
git clone https://github.com/leeqingming-BOL/mychinesemultipa.git
cd mychinesemultipa

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

We use the Mozilla CommonVoice dataset for Chinese. To prepare the data:

```bash
python preprocess.py \
      --output_dir data_new \
      --num_proc 48 \
      --cache_dir cache
```

This script:
- Loads the Chinese dataset from CommonVoice
- Converts Chinese text to IPA using our custom Chinese2IPA converter
- Prepares the data for training

## Model Training

To train the model:

```bash
python main.py \
       --train_samples 1000 \
       --test_samples 200 \
       --quality_filter True \
       --suffix=-zh-ipa1000 \
       --no_space True \
       --vocab_file vocab_zh.json \
       --data_dir data_new/ \
       --num_train_epochs 10 \
       --num_proc 48
```

Key parameters:
- `train_samples`: Number of training samples to use
- `test_samples`: Number of test samples
- `quality_filter`: Whether to filter low-quality audio samples
- `suffix`: Model name suffix for identification
- `no_space`: Whether to remove spaces from IPA transcriptions
- `num_train_epochs`: Number of training epochs

## Evaluation

To evaluate the model:

```bash
python test.py
```

This script:
- Loads a trained model
- Processes test samples
- Calculates various metrics (CER, PER, PFER, etc.)
- Outputs detailed results to CSV files

## Technical Methodology

### Multidimensional Data Construction

1. **Speech Data Source**: We select high-quality Mandarin Chinese speech segments from the CommonVoice dataset.

2. **Hierarchical Generation of IPA Sequences**:
   - The `pypinyin` tool converts text to standard pinyin
   - Our custom `Chinese2IPA` tool further converts pinyin to IPA sequences

### Model Architecture

Our model is based on the pretrained `wav2vec2-large-xlsr-53` framework, with specific modifications:

1. We incorporate the Connectionist Temporal Classification (CTC) loss function
2. We implement a fine-tuning strategy that shifts the output target from multi-lingual phoneme prediction to Mandarin Chinese IPA sequence generation
3. The feature extractor is frozen during fine-tuning to maintain pretrained speech representations

## Results

Our Chinese-specific Speech2IPA model with 1,000 training samples achieved a PFER of 16.2%, outperforming:
- Allosaurus: 20.9% PFER
- Wav2Vec2Phoneme: 18.3% PFER

These results validate our approach of prioritizing data quality over quantity and demonstrate the effectiveness of our Chinese-specific architecture.

## Applications

1. **Academic Research**: Support for dialect recording → IPA transcription → database construction

2. **Clinical Applications**: Early identification of language developmental issues in children, reducing diagnostic delays

3. **Language Learning**: IPA-annotated pronunciation examples serve as valuable learning aids

## Project Structure

```
mychinesemultipa/
├── converter/
│   └── chinese_to_ipa.py       # Chinese text to IPA conversion tool
├── data_utils.py               # Data processing utilities
├── main.py                     # Training script
├── preprocess.py               # Data preprocessing script
├── test.py                     # Evaluation script
├── utils.py                    # Utility functions and metrics
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Citation

If you use this model in your research, please cite:

```
@inproceedings{li2025pulse,
  title={Pulse: Chinese Speech2IPA with High-Quality Annotation},
  author={Li, Qingming and Wang, Youran and Sun, Ruiyan},
  booktitle={Proceedings of the CSC5051 Natural Language Processing Course},
  year={2025},
  organization={The Chinese University of Hong Kong, Shenzhen}
}
```

## Authors

- Qingming Li (224040228@link.cuhk.edu.cn)
- Youran Wang (224040259@link.cuhk.edu.cn)
- Ruiyan Sun (224040284@link.cuhk.edu.cn)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- We thank the Prof. Benyou WANG and TAs of the CSC5051 Natural Language Processing course at CUHK-SZ
- This research is inspired by the universal automatic phonetic transcription work by Taguchi et al. (2023)