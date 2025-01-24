from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration

modelcard = 'plenz/GLM-flan-t5-large'  
modelcard_generation = 'google/flan-t5-large'  

print('load the model and tokenizer')
model_generation = T5ForConditionalGeneration.from_pretrained(modelcard_generation)
del model_generation.encoder  # we only need the decoder for generation. Deleting the encoder is optional, but saves memory.
model = AutoModel.from_pretrained(modelcard, trust_remote_code=True, revision='main')
tokenizer = AutoTokenizer.from_pretrained(modelcard)
model_generation.shared = model.shared  # share embeddings between encoder and decoder. This mimics the T5 architecture.

print('get dummy input (2 instances to show batching)')
graph_1 = [
    ('black poodle', 'is a', 'dog'),
    ('dog', 'is a', 'animal'),
    ('cat', 'is a', 'animal')
]
text_1 = 'summarize: The black poodle chased the cat.'  # with T5 prefix

graph_2 = [
    ('dog', 'is a', 'animal'),
    ('dog', 'has', 'tail'),
    ('dog', 'has', 'fur'),
    ('fish', 'is a', 'animal'),
    ('fish', 'has', 'scales')
]
text_2 = "Dogs have <extra_id_0> and fish have <extra_id_1>. Both are <extra_id_2>."  # T5 MLM

print('prepare model inputs')
how = 'global'  # can be 'global' or 'local', depending on whether the local or global GLM should be used. See paper for more details. 
data_1 = model.data_processor.encode_graph(tokenizer=tokenizer, g=graph_1, text=text_1, how=how)
data_2 = model.data_processor.encode_graph(tokenizer=tokenizer, g=graph_2, text=text_2, how=how)
datas = [data_1, data_2]
model_inputs, attention_mask = model.data_processor.to_batch(data_instances=datas, tokenizer=tokenizer, max_seq_len=None, device='cpu', return_attention_mask=True)

print('compute token encodings')
outputs = model(**model_inputs)

print('generate conditional on encoded graph and text')
outputs = model_generation.generate(encoder_outputs=outputs, max_new_tokens=10, attention_mask=attention_mask)
print('generation 1:', tokenizer.decode(outputs[0], skip_special_tokens=True))
print('generation 2:', tokenizer.decode(outputs[1], skip_special_tokens=False))
