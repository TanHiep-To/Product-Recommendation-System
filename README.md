# Product Recommendation System

This project implements various recommendation systems for movies and beauty products using different filtering techniques. The system helps businesses improve their shoppers' experience and results in better customer acquisition and retention.

## Project Structure

```
Product-Recommendation-System/
├── Movie/
│   ├── data/
│   ├── notebooks/
│   │   ├── preprocessing.ipynb
│   │   ├── graphic_filter.ipynb
│   │   └── content_based_filtering.ipynb
│   └── pre_processing.py
├── Beauty/
│   ├── data/
│   ├── notebook/
│   │   └── data_pre_processing.ipynb
│   ├── scripts/
│   │   └── download.sh
│   └── main.py
└── README.md
```

## Features

- **Movie Recommendations**:
  - Demographic Filtering
  - Content-Based Filtering
  - Processing of TMDB 5000 movie dataset

- **Beauty Product Recommendations**:
  - Collaborative Filtering
  - Matrix Factorization using SVD
  - Processing of Amazon Beauty product ratings

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Jupyter Notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Product-Recommendation-System.git
cd Product-Recommendation-System
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Movie Recommendations

1. Run preprocessing:
```python
python Movie/pre_processing.py
```

2. Open Jupyter notebooks in `Movie/notebooks/` for different filtering techniques:
- `preprocessing.ipynb`: Data preprocessing steps
- `graphic_filter.ipynb`: Demographic filtering
- `content_based_filtering.ipynb`: Content-based recommendations

### Beauty Product Recommendations

1. Download the dataset:
```bash
cd Beauty/scripts
bash download.sh
```

2. Run the main recommendation system:
```python
python Beauty/main.py
```

## Data Sources

- Movies: TMDB 5000 Movie Dataset
- Beauty: Amazon Beauty Products Ratings Dataset

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/Product-Recommendation-System
