# Query

1. 기본적인 query를 사용하는 방법
    1. market price 관련 질문
        1. 질문이 어떤 코인에 대한 질문인지 파악 (특정 코인인지, 아니면 모든 코인에 대한 질문인지)
        2. 질문이 어떤걸 물어보는지 파악 (가격 변화를 물어보나? 거래량을 보나? 기술적 전략의 결과를 보나? 다이버전스를 보나?)
            1. 가격, 거래량 관련 “거래” 관련 질문 → candles 에서 쿼리해서 나온 결과를 prompt에 추가
            2. 기술적 전략의 결과를 본다 → indicator_signal_summary 에서 쿼리해서 나온 결과를 prompt에 추가
    2. 전략 자체에 관련 질문
        1. 특정 코인에 대한 질문이 아닌, 매매할 때 사용할 수 있는 전략에 대해서 물어본다.
        2. indicator_rules 쿼리해서 나온 결과를 prompt에 추가
    3. news 관련 질문
        1. 질문에 특정 날짜, 범위가 있는지 파악
        2. 질문에 어떤 구체적인 코인에 대한 질문인지 파악 (전반적인 코인 시장? 아니면 특정 코인)
        3. 위 두 조건을 넣은 쿼리를 news_articles 테이블을 쿼리해서 prompt에 추가
    4. multiple / general 질문
        1. 질문이 어떤 코인에 대한 질문인지 파악
        2. candles, indicator_signal_summary, news 에 대한 쿼리 결과를 모두 prompt에 넣어서 반환해야됨.

# Prompting

1. Prompt chaining
    1. general 한 question이 나오면 prompt chaining을 쓸 수 있게 해야된다.
    2. Is BTC attractive right now?
        1. Get btc candle data
        2. Get indicator_signal_summary information and find the overall_signal and dominant_ratio by time
        3. Find recent news about BTC and count the positive / negative news
        4. Reply to prompt in terms of technical indicator and news
    3. 이런식으로 prompt chaining이 되어야 한다.
2. Meta prompting
    1. 이건 정확하게 뭘 하는지 잘 모르겠음.
3. Self-reflection prompting
    1. 항상 답변하기 전에, 자기 내용을 다시 확인하고, 모순이 없는지 확인한다.
        1. answer 를 먼저 다시 확인한 다음, answer 내에 상반되는 내용이 없는지 확인.
        2. Hallucination이 일어났는지 확인.
        3. question - answer를 cache로 저장해서 비슷한 question에 대한 answer도 비슷했는지 확인? → 효율적인지 잘 모르겠음. 비슷한 질문이 없었으면?

# Computing

1. Prompt caching
    1. question - answer 를 저장하거나, query 결과를 어떻게 잘 indexing 해서 저장해서, prompt_caching 옵션이 켜지면 그 쿼리 결과를 바로 이용할 수 있게 했으면 좋겠다.
2. Student teacher model → 시간 되면 작은 모델 재학습 시켜보기

# Security Testing

- 이건 Prompting에 잘 녹여 넣어야 됨.