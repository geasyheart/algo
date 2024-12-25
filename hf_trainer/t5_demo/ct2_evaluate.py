import ctranslate2
from transformers import T5Tokenizer

translator = ctranslate2.Translator('subject-ct2')
tokenizer = T5Tokenizer.from_pretrained('checkpoint-68000')

for q in (
        '今天天气怎么样？', '今天中午吃什么？', '为什么会下雨', '黄晓明和angleboby什么时候结婚',
        '水杯超过100度是不是会爆炸',
        '什么是防洪调度', '“红豆生南国”中的南国在哪里', '宝宝什么时候可以不用喝夜奶？'):
    question = '主题提取，每个主题4个字：\n' + q

    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(question))
    results = translator.translate_batch(
        [input_tokens], max_decoding_length=60,
        # beam_size=2,
        sampling_topp=0.95,
        sampling_temperature=0.3,

    )
    output_tokens = results[0].hypotheses[0]
    output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens),
                                   skip_special_tokens=True)

    print(output_text)
