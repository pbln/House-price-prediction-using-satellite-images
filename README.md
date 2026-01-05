# House-price-prediction-using-satellite-images
# House Price Prediction using Tabular + Satellite Images

This project is an attempt to predict house prices by **combining structured property data with satellite images** of the surrounding area. The main idea is that house prices are influenced not only by numeric attributes (like size, location coordinates, etc.) but also by **visual context** — greenery, road density, urban layout, and neighborhood patterns visible from above.
The project started as a standard tabular ML pipeline and was later extended into a **multimodal setup** using CNN-based image embeddings.

---

## What this project does

* Builds a **baseline house price model** using tabular data only
* Fetches **satellite images** using latitude & longitude
* Extracts visual features from images using a **pretrained CNN (ConvNeXt)**
* Reduces image feature dimensionality using **PCA**
* Fuses tabular + image features
* Trains a **LightGBM regressor** on the fused feature space

---

## Repository structure

```
.
├── data_fetch.py              # Fetch satellite images from lat/long
├── EDA_image.ipynb            # Visual EDA on satellite images
├── preprocessing.ipynb   # Tabular-only LightGBM baseline
├── Model_training.ipynb       # Multimodal training & prediction
├── images/
│   ├── train/                 # Training satellite images
│   └── test/                  # Test satellite images
├── README.md

```

---

## Modeling approach

### 1. Tabular baseline
A LightGBM regressor is trained using only tabular features. This serves as a reference point to understand how much value image information adds.
### 2. Image feature extraction
* Satellite images are fetched using property latitude and longitude
* A pretrained **ConvNeXt** model is used as a **fixed feature extractor**
* No fine-tuning is performed; only embeddings are extracted
### 3. Dimensionality reduction
The raw CNN embeddings are high dimensional. To make them usable:
* **PCA is fit only on training image embeddings**
* The same PCA transformation is applied to test embeddings
### 4. Multimodal fusion
* Image embeddings are used as image features.
* PCA is used to reduce the dimension of the image features.
* Both image features and tabular features are then concatinated. 
### 5. Final model
A LightGBM regressor is trained on the fused feature space:
* Tabular features
* PCA-reduced image embeddings
---

## Results (high-level)
* Tabular-only model performs strongly due to structured signals , with RMSE of about 0.29 (for log prices) and R2 score of 0.69 (with all 16k images)
* Adding satellite image features provides **small but consistent improvements** , RMSE about 0.26 and R2 score of 0.73 (with about 6k images , due to time constraint)
* Image features capture locality-level signals that tabular data alone misses .

---

## Future improvements

* Fine-tune CNN on real-estate imagery
* Train on complete 16k images
* Fasten the image fetching proccess 
* Handle missing images using learned or mean embeddings
* Add spatial smoothing / neighborhood aggregation

---


