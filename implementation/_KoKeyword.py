import re
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

class _KoKeyword:
  """KoKeyword의 구현체

  KoKeyword의 내부 구현을 담은 클래스입니다.
  """
  def __init__(self, word_list):
    self.word_list = word_list # 입력받은 단어들의 데이터셋
    self.token_graph = {} # 토큰끼리 인접해서 등장한 횟수를 기록한 그래프
    self.word_to_tokens = {} # key: word, value: word가 쪼개져서 생성될 수 있는 토큰 리스트
    self.token_to_idx = {} # key: token, value: csr_matrix의 idx
    self.idx_to_token = {} # key: csr_matrix의 idx, value: token
    self.num_tokens = 0 # 주어진 데이터셋 내의 전체 토큰 수
    self.adj_csr_matrix = [] # 인접행렬(csr matrix) - 값: 인접해서 등장한 횟수
    self.rank = [] # 각 토큰의 최종 랭크
    self.length_frequency = {} # 길이 n인 토큰의 등장 횟수
    self.token_frequency = {} # 토큰별 등장 횟수 
    self.split_piece_list = {} # center token 양옆 부분을 본인 포함하여 저장하는 리스트


  def cleanse(self, word_list):
    """단어들에서 한국어만 남을 수 있도록 정제합니다.
    
    한국어 이외의 글자는 공백으로 변경합니다.

    Args:
      word_list (list(str)): 단어들 리스트
    
    Returns:
      list(str): 한국어만 남도록 정제된 단어들의 리스트 
    """
    korean_reg = re.compile('[^가-힣]')
    word_converted_list = []
    for word in word_list:
      # 한글 이외의 글자는 공백으로 변경
      word_converted_list.append(re.sub(korean_reg, ' ', word))
    return word_converted_list


  def split_words(self, word_list):
    """공백 기준으로 단어를 쪼갭니다.
    
    단어의 앞뒤 공백을 제거합니다.
    공백 묶음을 기준으로 단어를 잘라 리스트에 담아 반환합니다.

    Args:
      word_list (list(str)): 단어들 리스트
    
    Returns:
      list(list(str)): 공백 묶음을 기준으로 자른 단어들을 담은 리스트 
    """
    # 단어들을 공백 기준으로 쪼개기
    word_split_list = []
    for word in word_list:
      word_split_list.append(re.split(r'\s+', word.strip()))
    return word_split_list


  def make_graph(self, min_token_len=1):
    """인접한 토큰끼리 연결된 그래프를 생성해 반환합니다.

    min_token_len 이상의 길이를 가진 토큰으로 자릅니다.
    토큰의 앞뒤 토큰을 연결하는 엣지를 가지는 그래프를 만들어 리턴합니다.
    엣지의 가중치는 인접해서 등장한 횟수입니다.
    
    Args:
      min_token_len (int): 최소 토큰 길이
    
    Returns:
      dict(dict): {토큰A: {토크B: 토큰A와 토큰B가 인접해서 등장한 횟수}} 형태
    """
    # 한글, 영어가 아닌 특수문자는 공백으로 대체
    word_converted_list = self.cleanse(self.word_list)
    # 띄어쓰기 단위로 쪼개기
    word_split_list = self.split_words(word_converted_list)
    # 쪼개진 단어 하나하나 토큰화해서 그래프 만들기
    for words_split in word_split_list:
      for word_split in words_split:
        # 최초로 등장한 단어일 때만 저장
        is_appeared_first = False
        if word_split not in self.split_piece_list:
          is_appeared_first = True 
          self.split_piece_list[word_split] = [('', word_split, '')]
        # word_split 자체를 그래프에 넣기
        self.token_graph.setdefault(word_split, {})
        self.token_graph[word_split]['single'] = self.token_graph.get(word_split, {}).get('single', 0) + 1
        # 길이별 빈도수 기록
        self.length_frequency[len(word_split)] = self.length_frequency.get(len(word_split), 0) + 1
        # 토큰 빈도수 + 1
        self.token_frequency[word_split] = self.token_frequency.get(word_split, 0) + 1
        # 단어를 토큰으로 쪼개기
        for center_toekn_len in range(min_token_len, len(word_split)):
          for start in range(len(word_split) - center_toekn_len + 1):
            center_token = word_split[start:start+center_toekn_len]
            prev_token = word_split[:start]
            next_token = word_split[start+center_toekn_len:]
            # center_token을 길이별 빈도수에 반영
            self.length_frequency[len(center_token)] = self.length_frequency.get(len(center_token), 0) + 1
            # 토큰 빈도수 + 1
            self.token_frequency[center_token] = self.token_frequency.get(center_token, 0) + 1
            # 나중에 단어를 대표할 수 있는 토큰을 찾기 위해 단어에서 나올 수 있는 토큰들을 저장
            if is_appeared_first:
              self.split_piece_list[word_split].append( (prev_token, center_token, next_token) )
            # 중심 토큰과 앞뒤 토큰을 그래프 에지로 만들기
            # 맛있는꽃새우 : 맛있는 -> 꽃, 꽃 -> 맛있는 
            # center_token 기준으로 양쪽에 edge 연결 (양 끝의 토큰이면 한 방향에만 토큰이 존재할 수도 있음)
            # 같은 토큰끼리는 연결하지 않기
            if center_token not in self.token_graph:
              self.token_graph[center_token] = {}
              if prev_token and center_token != prev_token: 
                self.token_graph[center_token][prev_token] = 1
              if next_token and center_token != next_token:
                self.token_graph[center_token][next_token] = 1
            else:
              if prev_token and center_token != prev_token:
                self.token_graph[center_token][prev_token] = self.token_graph[center_token].get(prev_token, 0) + 1
              if next_token and center_token != next_token:
                self.token_graph[center_token][next_token] = self.token_graph[center_token].get(next_token, 0) + 1
    return self.token_graph


  def make_adj_csr_matrix(self):
    """빠른 행렬 연산을 위해 그래프를 csr_matrix로 변환합니다
    
    그래프를 csr_matrix 형태의 인접 행렬로 변환합니다.
    """
    self.token_to_idx = {token : idx for idx, token in enumerate(self.token_graph.keys())}
    self.idx_to_token = {idx : token for idx, token in enumerate(self.token_graph.keys())}
    self.num_tokens = len(self.token_to_idx.keys())
    # 끝에 'single'(단독으로 등장한 횟수) 추가
    self.token_to_idx['single'] = self.num_tokens
    # csr_matrix로 인접행렬 생성
    row_idx = [self.token_to_idx[token] for token, adj_token_dict in self.token_graph.items() for _ in range(len(adj_token_dict.keys()))]
    col_idx = [self.token_to_idx[adj_token] for token, adj_token_dict in self.token_graph.items() for adj_token, cnt in adj_token_dict.items()]
    data = [cnt for token, adj_token_dict in self.token_graph.items() for adj_token, cnt in adj_token_dict.items()]
    # 행, 열 끝에 'single' 추가
    self.adj_csr_matrix = csr_matrix((data, (row_idx, col_idx)), shape=(self.num_tokens+1, self.num_tokens+1))
    return self.adj_csr_matrix


  def compute_token_rank(self, single_word_weight_percentage=0.5, single_word_weight_count=1.0):
    """그래프를 가지고 토큰별 랭크를 계산합니다.

    각 토큰의 초기 가중치는 토큰 길이별 등장 빈도수입니다.
    토큰이 한 덩어리로 혼자 존재하는 경우에만 가중치를 다르게 적용합니다. 매개변수로 지정해줄 수 있습니다.

    Args:
      single_word_weight_percentage (float): 전달하는 데이터셋의 띄어쓰기를 믿는 정도 (인접해서 등장한 비율)
      single_word_weight_count (float): 전달하는 데이터셋의 띄어쓰기를 믿는 정도 (인접해서 등장한 횟수)
     """
    # 토큰 길이별 등장 빈도수
    u1 = [(self.token_frequency[token] / self.length_frequency[len(token)])   for token in self.token_graph.keys()]
    # 단독으로 등장하는 경우 정규화 결과 single만 1이 됨 - 반영 비율이 너무 큼
    u1.append(single_word_weight_percentage)

    # 토큰 길이별 등장 빈도수
    u2 = [(self.token_frequency[token] / self.length_frequency[len(token)])   for token in self.token_graph.keys()]
    # 단독으로 존재하는 경우
    u2.append(single_word_weight_count)

    # 인접해서 등장한 비율 이용
    A = normalize(self.adj_csr_matrix, axis=1, norm='l1')
    # 인접해서 등장한 횟수 이용
    B =  self.adj_csr_matrix

    self.rank =  A.dot(u1) * B.dot(u2)
    return self.rank


  def get_token_rank(self, top_n):
    """토큰 랭크를 얻기 위해 호출하는 인터페이스

    각 단어별 상위 토큰과 랭크를 반환합니다.

    Args:
      top_n (int): 각 단어별 가장 의미있는 토큰 몇 개를 추출할 지를 의미합니다.

    Returns:
      dict(list(tuple)): {'단어A': [ ('토큰B', 토큰B의 중요도), ('토큰C', 토큰C의 중요도), ... ]} 형태입니다.
    """
    self.make_graph()
    self.make_adj_csr_matrix()
    self.compute_token_rank()
    return self.get_token_rank_with_side_tokens(top_n)


  def get_token_rank_with_one(self, top_n):
    """토큰 본인의 랭크만 이용하여 각 단어에서 의미있는 토큰들을 추출합니다.

    주어진 단어 리스트에서 만든 그래프를 바탕으로 각 단어에서 의미있는 토큰 top_n개를 추출합니다.
    
    Args:
      top_n (int): 각 단어별 가장 의미있는 토큰 몇 개를 추출할 지를 의미합니다.

    Returns:
      dict(list(tuple)): {'단어A': [ ('토큰B', 토큰B의 중요도), ('토큰C', 토큰C의 중요도), ... ]} 형태입니다.
    """
    word_to_token_rank = {}
    for i, words_split in enumerate(self.split_words(self.cleanse(self.word_list))):
      token_rank = []
      for word in words_split:
        for prev_token, center_token, next_token in self.split_piece_list[word]:
          token_rank.append((center_token, self.rank[self.token_to_idx[center_token]]))
      token_rank = list(set(token_rank)) # 중복 제거
      token_rank.sort(key=lambda x: (-x[1]))
      word_to_token_rank[self.word_list[i]] = token_rank[:top_n] # 랭크 상위 토큰  n개


  def get_token_rank_with_side_tokens(self, top_n, beta=0.6):
    """토큰 본인의 랭크뿐만 아니라 양 옆 토큰의 랭크까지 고려하여 각 단어에서 의미있는 토큰들을 추출합니다.

    주어진 단어 리스트에서 만든 그래프를 바탕으로 각 단어에서 의미있는 토큰 top_n개를 추출합니다.
    주변 토큰의 맥락을 조금 더 고려하기 위해 양옆 토큰의 랭크를 함께 계산합니다.
    (토큰 본인의 랭크 = 양옆 토큰의 랭크의 평균 * beta + 토큰 본인의 랭크) 계산식을 이용해 랭크를 재계산합니다.

    Args:
      top_n (int): 각 단어별 가장 의미있는 토큰 몇 개를 추출할 지를 의미합니다.
      beta (float): 양옆 토큰의 랭크를 얼마나 반영할 지를 의미합니다.

    Returns:
      dict(list(tuple)): {'단어A': [ ('토큰B', 토큰B의 중요도), ('토큰C', 토큰C의 중요도), ... ]} 형태입니다.
    """
    word_to_token_rank = {}
    for i, words_split in enumerate(self.split_words(self.cleanse(self.word_list))):
      token_rank = []
      for word in words_split:
        for prev_token, center_token, next_token in self.split_piece_list[word]:
          side_n = 0 # 양옆에 존재하는 토큰 개수 (0, 1, 2)
          if not prev_token:
            side_n += 1
            left_rank = 0
          else:
            side_n += 1
            left_rank = self.rank[self.token_to_idx[prev_token]]
          if not next_token:
            right_rank = 0
          else:
            right_rank = self.rank[self.token_to_idx[next_token]]
          # 주변에 토큰이 없으면 left_rank, right_rank 모두 0
          # 0으로 나누지 않게 처리
          if side_n == 0:
            side_n = 1
          # 주변 토큰 랭크의 평균 * beta + 토큰 본인의 랭크
          token_rank.append((center_token, (left_rank + right_rank) / side_n * beta + self.rank[self.token_to_idx[center_token]] ))
      # 중복 토큰 제거- 가장 높은 랭크로 기록
      token_max_rank = {}
      for token, rank in token_rank:
        if token in token_max_rank:
          token_max_rank[token] = max(token_max_rank[token], rank)
        else:
          token_max_rank[token] = rank
      token_max_rank = sorted(list(token_max_rank.items()), key=lambda x: (-x[1]))
      word_to_token_rank[self.word_list[i]] = token_max_rank[:top_n] # 랭크 상위 토큰 n개만 추출
    return word_to_token_rank

