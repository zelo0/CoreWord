from .implementation._KoKeyword import _KoKeyword

class KoKeyword:
  """단어들의 리스트가 주어지면 빈도수를 기반으로 각 단어에서 의미있는 토큰들을 추출합니다.

  (예)닭안심살 -> 닭(1순위), 안심(2순위), ...

  띄어쓰기에 대한 신뢰를 바탕으로 학습이 이루어지는 모델입니다.
  띄어쓰기가 올바르게 된 정보를 바탕으로 띄어쓰기가 되지 않은 단어에서도 유의미한 토큰을 추출할 수 있습니다.
  길이별 빈도수, 인접한 토큰들 중 등장 비율, 인접한 토큰의 등장 횟수를 고려합니다.

  Args:
    word_list (list(str)): 학습할 단어들의 리스트
  """

  def __init__(self, word_list):
    self.word_list = word_list # 입력받은 단어들의 데이터셋

  def get_token_rank_for_each_word(self, top_n=3):
    """각 단어의 토큰들 중 순위가 높은 n개를 리턴해줍니다 (기본값 n = 3)
    
    Args:
      top_n (int): 단어별로 리턴받을 상위 토큰 개수

    Returns:
      dict(list(tuple)): {'단어A': [ ('토큰B', 토큰B의 중요도), ('토큰C', 토큰C의 중요도), ... ]} 형태입니다.
    """
    _kokeyword = _KoKeyword(self.word_list)
    return _kokeyword.get_token_rank(top_n)