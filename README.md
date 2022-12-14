# CoreWord
핵심 단어 찾기  
## 문제 정의
한국어로 적힌 문장들을 보면 띄어쓰기가 안 되어있는 경우가 많습니다. 인터넷에 있는 문장에서 유독 이러한 경향을 보입니다.  띄어쓰기가 안 돼있는 경우에도 유의미한 단어를 찾아낼 수 있어야 합니다. 기존의 konlpy 토크나이저는 미리 학습된 데이터를 바탕으로 단어를 찾습니다. 그렇다보니 신조어나 학습되지 않은 단어를 잘 차지 못하는 경향이 있습니다. 하나의 도메인에서 나온 단어들만을 가지고 학습하게 된다면 유의미한 단어를 더 잘 찾을 수 있을 것입니다.


레시피 추천을 위해 레시피에 있는 재료들의 단어를 정제할 필요가 있었습니다. 개인이 올린 재료명은 다양한 형태를 가집니다.  
재료명은 만개의 레시피에서 가져왔습니다. 아래는 소금이라는 단어가 들어간 재료명입니다. 
* ['소금코지 또는 소금', '소금', '맛소금', '깨소금', '꽃소금', '허브소금', '굵은소금', '볶음 깨소금', '소금약간', '오이 절일 소금', '굵은 소금', '깨소금(볶은통깨간것)', '소금(선택)', '고운 소금', '무절임-소금', '허브솔트(후추+소금)', '야채데칠때 소금', '고운소금', '소금물', '데치는 소금', '볶은통깨 (참깨)또는 꺠소금', '소금 2g ×', '가는 소금', '소금(볶음소금)', '통깨 or 깨소금', '볶은소금', ... ]


미묘하게 차이가 있겠지만 저는 '굵은 소금', '꽃소금', '깨소금', '고운 소금'이 전부 '소금'이라는 하나로 묶기를 원했습니다. 단순히 토크나이저를 사용해서 품사 태깅을 하고 명사만 뽑으려는 생각을 했었습니다. '3mm두께 돼지목살'이라는 재료명도 올라와있습니다. 이 경우에 토크나이저가 명사를 잘 뽑으면 '두께', '돼지', '목살'을 추출할 것입니다. 그렇다면 토크나이저만 사용했을 때 어떤 게 더 중요하다고 말할 수 있습니까?  
재료라는 도메인을 고려하면 전체 데이터셋에서 많이 나온 '돼지'나 '목살'이 더 중요하다고 말할 수 있습니다. 토크나이저와 빈도수 계산을 통해 어느정도 성능을 낼 수 있을 것으로 추정됩니다. 하지만 문제가 있습니다. 학습되지 않은 재료들이 있습니다.

konlpy가 제대로 학습하지 못한 단어로는 스리라차, 두반장, 매실액 등이 있습니다.

![매실액](https://user-images.githubusercontent.com/86035717/190895963-e4262043-303f-4ca9-a4e0-a9091cb80eed.png)

![두반장](https://user-images.githubusercontent.com/86035717/190895274-0fc339b2-6f12-4f35-8d10-64dc326194b4.png)

기존에 학습되지 않은 재료를 위해서 핵심 토큰 추출기를 만들었습니다.





## 아이디어
재료 데이터셋을 주면 이 데이터셋 내에서 많이 나오는 토큰은 중요한 토큰일 확률이 높습니다. '달걀'이나 '쌀' 등이 그 예입니다. '쌀'이 들어간 단어에서는 '쌀'이 중요한 토큰일 수 가능성이 높습니다. 이 모델은 빈도수를 기반으로 만들었습니다.  
1. 모든 단어들에서 한글만 남기고 공백을 기준으로 자릅니다.
2. 모든 단어들을 길이별로 슬라이싱합니다.
3. 서로 인접해서 등장한 토큰끼리 엣지로 연결하는 그래프를 생성합니다. 엣지의 가중치는 인접해서 등장한 횟수입니다.
4. 인접 행렬 내에서 각 토큰의 랭크를 계산합니다.
5. 단어별로 인접한 토큰의 랭크를 반영하여 토큰별 최종 랭크를 계산합니다.




## 장점
konlpy의 토크나이저와 빈도수를 사용했을 경우보다 못 한 케이스도 분명 존재합니다. 여기에서는 konlpy가 잡지 못한 부분을 잡은 예만 골라봤습니다.


매실액의 경우 '매실액', '매실', '액' 순으로 의미 있는 토큰이라는 결과가 나왔습니다.
![매실액](https://user-images.githubusercontent.com/86035717/190896081-20f5ba1f-b0cf-4ce5-94d2-13b2a32ce801.png)  
두반장의 경우 '두반장', '반', '장' 순으로 의미 있는 토큰이라는 결과가 나왔습니다.
![두반장](https://user-images.githubusercontent.com/86035717/190896136-a1ba666d-6e61-46da-8ce5-2f18e2d3211f.png)

