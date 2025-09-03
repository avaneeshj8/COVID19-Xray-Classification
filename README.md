

# COVID-19 Chest X-ray Classification using CNN

## Project Overview
This project focuses on **classifying chest X-ray images into COVID-19 positive and Normal cases** using a **Convolutional Neural Network (CNN)**.  
Early and accurate detection of COVID-19 is critical for treatment and control of the pandemic. Chest radiography is widely available and inexpensive, making deep learning a powerful tool for rapid screening.  

Our approach uses a custom CNN trained on the **COVID-19 Radiography Dataset**, achieving strong classification performance between infected and normal samples.

---

## Dataset
- **Source**: [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)  
- **Classes**:  
  - COVID-19 (labeled as `1`)  
  - Normal (labeled as `0`)  
- **Preprocessing**:  
  - Images resized to **100×100**  
  - Pixel values normalized to **[0,1]**  
  - Dataset balanced by randomly sampling equal numbers of Normal and COVID images  

---

## Model Architecture
The CNN was built using **TensorFlow/Keras** with the following layers:



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">98</span>, <span style="color: #00af00; text-decoration-color: #00af00">98</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │           <span style="color: #00af00; text-decoration-color: #00af00">896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">47</span>, <span style="color: #00af00; text-decoration-color: #00af00">47</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">4,624</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">23</span>, <span style="color: #00af00; text-decoration-color: #00af00">23</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">2,320</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1600</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">819,712</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">131,328</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">257</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>

</details>

---

## Training
- Optimizer: **Adam**  
- Loss: **Binary Crossentropy**  
- Metric: **Accuracy**  
- Early stopping applied (patience = 3)  

---

## Results

### Accuracy over Epochs
<img width="576" height="432" alt="accuracy plot" src="https://github.com/user-attachments/assets/b4d109c0-d738-4e84-919c-03d9b143db43" />

### Loss over Epochs
<img width="576" height="432" alt="loss plot" src="https://github.com/user-attachments/assets/56e83dfc-ab55-491a-93b7-9f0ea61315e9" />

### Confusion Matrix
The confusion matrix below shows model predictions on the test set, highlighting its ability to distinguish between COVID and Normal cases.

---
|                   | Predicted Normal | Predicted COVID |
| ----------------- | ---------------- | --------------- |
| **Actual Normal** | 93.8%            | 6.2%            |
| **Actual COVID**  | 10.4%            | 89.6%           |

## Conclusion

This project demonstrates that a Convolutional Neural Network can effectively classify COVID-19 and Normal chest X-ray images with high accuracy (~94%). By successfully distinguishing between infected and healthy lungs, the model shows promise as a rapid screening tool. Key factors contributing to its performance include careful data preprocessing and balancing, which ensure robust results, and the ability of CNNs to capture complex features in medical images without manual feature extraction. While the model performs well on the current dataset, further validation on larger and more diverse datasets is necessary before clinical deployment. Overall, this study highlights the potential of deep learning in aiding early detection of COVID-19 and provides a foundation for developing more advanced medical imaging solutions.

