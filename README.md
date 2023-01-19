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
            <td rowspan=1>Preprocessing</td>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/k-fold/Stratified%20K-Fold.ipynb">Stratified K-Fold.ipynb</a>
            </td>
            <td> Startified K-Fold </td>
        </tr>
        <tr>
            <td rowspan=3>Train</td>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/BERT/bert.ipynb">bert.ipynb</a>     
            <td> BERT/RoBERTa/ELECTRA </td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/T5/t5_train.py">t5_train.py</a>
            <td> T5 </td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/MLP/mlp_train.py">mlp_train.py</a>
            <td> MLP </td>
        </tr>
        <tr>
            <td rowspan = 3>Evaluate</td>
            <td>
                <a href="https://github.com/BBaekdabang/EmotionClassification/blob/main/Inference.ipynb">Inference.ipynb</a>     
            <td> BERT/RoBERTa/ELECTRA </td>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/T5/t5_evaluate.py">t5_evaluate.py</a>
            <td> T5 </td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/MLP/mlp_evaluate.py">mlp_evaluate.py</a>
            <td> MLP </td>
        </tr>
        </tr>        
        <tr>
            <td rowspan = 2>Ensemble</td>       
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/ensemble/hardvoting.py">hardvoting.py</a>
            <td> Hard Voting</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/ensemble/softvoting.py">softvoting.py</a>
            <td> Soft Voting</td>
        </tr>

   </tbody>
</table>

---

## 가. Train(ClovaAI)

    !CUDA_VISIBLE_DEVICES=0 python3 ./train.py \
    --train_data train_lmdb \
    --valid_data val_lmdb \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --batch_size 150 --data_filtering_off --workers 0 --imgH 64 --imgW 173 \
    --num_iter 180000 --valInterval 500 \
    # --saved_model best_accuracy.pth


## 나. Inference(ClovaAI)

    !touch infe.txt

    !CUDA_VISIBLE_DEVICE=0,1,2,3 python3 ./Inference.py \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --image_folder ./test/ \
    --log_filename ./infe.txt \ #Inference 결과값 저장되는 파일
    --saved_model ./best_accuracy.pth
    
## 다. Preprocessing

### ImageBinarization

<img width="1000" img height="250" src="https://user-images.githubusercontent.com/113493692/213394717-56e6d40e-0b5f-4c2a-a97d-dbca281ddb97.png">

### SuperResolution

<img width="1000" img height="250" src="https://user-images.githubusercontent.com/113493692/213395275-e96b4dfd-b8a9-430b-8cd7-69beead9331e.png">

---

