# OCR_TextRecognitionModel
CRNN, ClovaAI, Preprocessing

## Optical Character Recognition
<img width="200" img height="140" src="https://user-images.githubusercontent.com/113493692/213380584-1559b248-8ffc-47ea-8e5b-711a9b605ae3.png"> <img width="200" img height="140" src="https://user-images.githubusercontent.com/113493692/213381326-4b452736-cbc6-4b2c-8ce4-e5fcf3d70147.png"> <img width="200" img height="140" src="https://user-images.githubusercontent.com/113493692/213381457-2b4e166f-7820-4288-8046-3bf1195407b1.png"> <img width="200" img height="140" src="https://user-images.githubusercontent.com/113493692/213381545-0afe3f25-b4c2-415b-9b84-52beabd65e96.png">

### Dataset : [Dataset](https://dacon.io/competitions/official/236042/data)


---
## CODE

<table>
    <thead>
        <tr>
            <th>목록</th>
            <th>파일명</th>
            <th>설명</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>preprocessing</td>
            <td>
                <a href="https://github.com/BBaekdabang/OCR_TextRecognitionModel/blob/main/preprocessing/preprocessing.ipynb">preprocessing.ipynb</a>
            </td>
            <td> Binarization, Diltion, Erosion </td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/OCR_TextRecognitionModel/blob/main/preprocessing/SuperResolution.ipynb">SuperResolution.ipynb</a>
            </td>
            <td> Super Resolution </td>
        </tr>
        <tr>
            <td rowspan=2>train</td>
            <td>
                <a href="https://github.com/BBaekdabang/OCR_TextRecognitionModel/blob/main/ClovaAI/train.py">train.py</a>     
            <td> ClovaAI </td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/OCR_TextRecognitionModel/blob/main/baseline.ipynb">baseline.ipynb</a>
            <td> DACON baseline </td>
        </tr>
        <tr>
            <td rowspan = 1>evaluate</td>
            <td>
                <a href="https://github.com/BBaekdabang/OCR_TextRecognitionModel/blob/main/ClovaAI/inference.py">inference.py</a>     
            <td> ClovaAI </td>
        <tr>
        </tr>


   </tbody>
</table>

---

## 가. Train(ClovaAI)

    !CUDA_VISIBLE_DEVICES=0 python3 ./OCR_TextRecognitionModel/ClvoaAI/train.py \
    --train_data train_lmdb \
    --valid_data val_lmdb \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --batch_size 150 --data_filtering_off --workers 0 --imgH 64 --imgW 173 \
    --num_iter 180000 --valInterval 500 \
    # --saved_model best_accuracy.pth


## 나. Inference(ClovaAI)

    !touch infe.txt

    !CUDA_VISIBLE_DEVICE=0,1,2,3 python3 ./OCR_TextRecognitionModel/ClvoaAI/inference.py \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --image_folder ./test/ \
    --log_filename ./infe.txt \ #Inference 결과값 저장되는 파일
    --saved_model ./best_accuracy.pth
 
## 다. Preprocessing

### ImageBinarization

<img width="1000" img height="250" src="https://user-images.githubusercontent.com/113493692/213394717-56e6d40e-0b5f-4c2a-a97d-dbca281ddb97.png">

### SuperResolution

<img width="1000" img height="220" src="https://user-images.githubusercontent.com/113493692/213395275-e96b4dfd-b8a9-430b-8cd7-69beead9331e.png">

### Dilation

<img width="500" img height="250" src="https://user-images.githubusercontent.com/113493692/213412756-1dfbeb0c-d89c-45f0-aeb2-1906c43d790e.png">

### Erosion

<img width="500" img height="250" src="https://user-images.githubusercontent.com/113493692/213412861-963a220a-7c57-408f-a944-addf45034f1e.png">

---

## 라. Results

- ClovaAI (TPS / VGG / None / Attention)

<img width="800" img height="600" src="https://user-images.githubusercontent.com/113493692/214503742-a3b7dec4-1899-4130-82b2-0fc5523fe5fa.png">

## 바. Discussion
 
 -
 - 
## 바. Reference

[1] [ClovaAI](https://github.com/clovaai/deep-text-recognition-benchmark) : What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis

[2] [SuperResolution](https://github.com/JingyunLiang/SwinIR) : SwinIR: Image Restoration Using Swin Transformer

[3] [DACON](https://dacon.io/competitions/official/236042/codeshare/7345?page=1&dtype=recent) : [Baseline] CRNN(Resnet + RNN) + CTC Loss
