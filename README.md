## Aimers TST 모델 구현
model directory는 Time-Series Transformer(TST) 모델을 구현한 디렉토리며, 데이터로드 및 학습, 평가까지 이루어질 수 있다.
### 파일구조
 * [data_loder.py](./data_loader.py)
 * [utils.py](./utils.py)
 * [model.py](./model.py)
 * [eval.py](./eval.py)
 * [train.py](./train.py)
 * [config.py](./config.py)
 * [README.md](./README.md)
----
- `data_loader.py: 데이터 처리 파일`
- `utils.py: 학습 및 평가에 필요한 유틸리티 함수 포함 파일`
- `model.py: 모델 구현 파일(TST model)`
- `eval.py: 모델을 평가하기 위한 파일`
- `train.py: 모델을 학습하기 위한 파일`
- `config.py: 모델을 학습하기 위한 파라미터`
- `README.md: 리드미 파일`


### 실행방법
- config.py: 파라미터를 여기서 수정가능하다.
- train.py: 파일을 실행하여 모델을 학습시킨다. train.py 내 TODO에서 찾아 데이터를 적절히 불러온다.
- eval.py: 파일을 실행하여 모델의 성능평가를 진행한다.