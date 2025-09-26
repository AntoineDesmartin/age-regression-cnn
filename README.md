# Age Regression CNN — UTKFace (TensorFlow + Gradio)

Estimate a person's **age (régression continue)** from a **face image** using transfer learning
(**MobileNetV2** as feature extractor + regression head).  
Dataset: **UTKFace** (public) loaded via **`kagglehub`**.

**The trained weights are available on my [Hugging Face](https://huggingface.co/tonioexe/age-regression-cnn) or models/age_regressor.h5**

> ⚠️ Educational project. Age estimation from faces can be **biased and inaccurate** depending on lighting, pose, demographics, etc. Use responsibly.

---

## 🧪 Local demo (Gradio)

You can launch a small Gradio app that loads a saved model file (`models/age_regressor.h5`).

```bash
# 1) Clone repo
      git clone https://github.com/AntoineDesmartin/age-regression-cnn.git

      cd age-regression-cnn

# 2) Create a venv and install deps
python -m venv .venv

source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt

# 3) Run
python app.py
```

Then open the printed local URL in your browser.

---

## 🚀 Quickstart (Colab)

1. Open the notebook: `notebooks/Age_Regression_CNN_UTKFace.ipynb` in Google Colab.
2. Run all cells. It will:
   - download UTKFace via `kagglehub`,
   - build & train the model (transfer learning),
   - report MAE and ±5-year accuracy,
   - optionally save `age_regressor.h5`,
   - launch a **Gradio** demo.


---


## 📁 Repo structure

```
age-regression-cnn/
├── app.py                     
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   └── Age_Regression_CNN_UTKFace.ipynb
├── models/
    └── age_regressor.h5
```


---

## 📊 Results


After training and light fine-tuning, the performance on the test set is:

- **MSE** (*Mean Squared Error*): `35.8`  
- **MAE** (*Mean Absolute Error*): `4.43 years`  
- **Accuracy within ±5 years**: `64.5%`

### 📝 Interpretation
- On average, the model is off by about **4.4 years**.  
- In roughly **2 out of 3 cases**, the predicted age is within ±5 years of the true age.  
- Given the variability in the UTKFace dataset (lighting, expressions, demographics), these results are considered **solid for an educational demo**.

⚠️ **Note**: This project is intended for **educational purposes only**. Age estimation from faces is inherently approximate and may reflect dataset biases.


---

## 🧱 Model
- Backbone: `MobileNetV2` (ImageNet, frozen then partial fine-tune)
- Head: GAP → Dense(128, ReLU) → Dropout → Dense(1, linear)
- Preprocessing: resize (160×160), MobileNetV2 preprocessing
- Loss: MSE; Metric: MAE

---

## 🔐 Ethics & Bias
- Face-based age estimation can encode demographic and dataset biases.
- Do not deploy in sensitive/real-world decision systems without rigorous evaluation and approvals.

---

## 📜 License
UTKFace dataset usage is subject to its own license/terms — please review Kaggle's dataset page.
