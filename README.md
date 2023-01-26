# nanoGPT shakespeare
### using Google Colab to finetune nanoGPT on shakespeare

* [Based on karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
* [Example Jupyter Notebook on GitHub](https://github.com/eniompw/nanoGPTshakespeare/blob/main/nanoGPTshakespeare.ipynb)
* [Example on Colab](https://colab.research.google.com/drive/1G97dn-Ivle2PgjH3MXjnkOHYOnxlrf79)

train.py arguments explained:

* colab GPU doesn't support default bfloat16
  * `--dtype=float32`
* colab currently uses PyTorch 1.13.1+cu116 but compile requires PyTorch 2.0
  * `--compile=False`
*  largest GPT that seems to work on Colab
   *  `--init_from=gpt2-medium`
* ["smaller Transformer"](https://github.com/karpathy/nanoGPT#i-only-have-a-macbook)
  * `--n_layer=4 --n_head=4 --n_embd=64 block_size=64 --batch_size=8`
* save model every 100 iters:
  * `--eval_interval=100`


**Colab Code:**
```
  # download repo
  !git clone https://github.com/karpathy/nanoGPT.git
  
  # install dependencies
  pip install tiktoken transformers
  
  # download shakespeare dataset into ./data/shakespeare
  !cd /content/nanoGPT/data/shakespeare/ && python prepare.py
  
  # finetune gpt-medium with "smaller Transformer" on GPU, model in ./out. (200 iters seems to have lowest val loss) 
  !cd /content/nanoGPT/ && python train.py --dataset=shakespeare --n_layer=4 --n_head=4 --n_embd=64 --compile=False --block_size=64 --batch_size=8 --init_from=gpt2-medium --dtype=float32 --eval_interval=100
  
  # print 5 samples, with 5 tokens, starting with "to be"
  !cd /content/nanoGPT && python sample.py --dtype=float32 --num_samples=5 --max_new_tokens=5 --start="to be"
```
