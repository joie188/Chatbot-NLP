# Chatbot-NLP
6.806/864 Advanced NLP Final Project

1. Clone repo and original ParlAI repo 
2. Copy ju2jo.py into ParlAI/parlai/agents/examples folder 
3. Go into ParlAI folder and run python setup.py develop
4. Run `parlai train_model --model examples/ju2jo --model-file /tmp/example_model --task personachat --batchsize 32 --num-epochs 2`

To evaluate: `parlai eval_model --task personachat --model-file ./ju2jo_model --model examples/ju2jo --metrics accuracy,hits@1,token_acc,ppl,f1`
To generate samples: `parlai display_model --task personachat --model-file ./ju2jo_model --model examples/ju2jo --datatype valid`
