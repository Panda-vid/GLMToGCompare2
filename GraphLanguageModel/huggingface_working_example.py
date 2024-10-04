from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration

modelcard = 'plenz/GLM-t5-large'  
modelcard_generation = 't5-large' 

print('load the model and tokenizer')
model_generation = T5ForConditionalGeneration.from_pretrained(modelcard_generation)
del model_generation.encoder  # we only need the decoder for generation. Deleting the encoder is optional, but saves memory.
model = AutoModel.from_pretrained(modelcard, trust_remote_code=True, revision='main')
tokenizer = AutoTokenizer.from_pretrained(modelcard)


print('get dummy input (2 instances to show batching)')
graph_1 = [
    ('Jennings Randolph Bridge', 'crosses', '<extra_id_0>')
]
text_1 = "Over which river does Jennings Randolph Bridge cross?"

graph_2 = [
    ('dog', 'is a', 'animal'),
    ('dog', '<extra_id_0>', 'tail'),
    ('dog', 'has', 'fur'),
    ('fish', 'is a', 'animal'),
    ('fish', 'has', 'fins')
]
text_2 = "Dogs have tails and fish have fins. Both are animals."  # T5 MLM

print('prepare model inputs')
how = 'global'  # can be 'global' or 'local', depending on whether the local or global GLM should be used. See paper for more details. 
data_1 = model.data_processor.encode_graph(tokenizer=tokenizer, g=graph_1, text=text_1, how=how)
data_2 = model.data_processor.encode_graph(tokenizer=tokenizer, g=graph_2, text=text_2, how=how)
datas = [data_1, data_2]
model_inputs = model.data_processor.to_batch(data_instances=datas, tokenizer=tokenizer, max_seq_len=None, device='cpu')

print('compute token encodings')
outputs = model(**model_inputs)

print('generate conditional on encoded graph and text')
outputs = model_generation.generate(encoder_outputs=outputs, max_new_tokens=10)
print('generation 1:', tokenizer.decode(outputs[0], skip_special_tokens=True))
print('generation 2:', tokenizer.decode(outputs[1], skip_special_tokens=False))