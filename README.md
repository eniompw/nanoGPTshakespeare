# nanoGPT shakespeare
### using Google Colab to finetune nanoGPT on shakespeare

* [Example Jupyter Notebook on Colab](https://colab.research.google.com/drive/1G97dn-Ivle2PgjH3MXjnkOHYOnxlrf79)
* [Based on karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
* [Example Jupyter Notebook on GitHub](https://github.com/eniompw/nanoGPTshakespeare/blob/main/nanoGPTshakespeare.ipynb)


### Train: finetune GPT on the shakespere dataset  
`python train.py --dtype=float32 --dataset=shakespeare --compile=False --n_layer=4 --n_head=4 --n_embd=64 --block_size=64 --batch_size=8 --init_from=gpt2 --eval_interval=100 --eval_iters=100 --max_iters=300`

`train.py` arguments explained:

* colab GPU doesn't support default bfloat16
  * `--dtype=float32`
* colab currently uses PyTorch 1.13.1+cu116 but compile requires PyTorch 2.0
  * `--compile=False`
*  larger than `gpt2-medium` models run out of RAM (12.7GB) on Colab
   *  `--init_from=gpt2-medium`
* ["smaller Transformer"](https://github.com/karpathy/nanoGPT#i-only-have-a-macbook) speeds up training significantly 
  * `--n_layer=4 --n_head=4 --n_embd=64 block_size=64 --batch_size=8`
* save model every 100 iters:
  * `--eval_interval=100`
* calculate val loss for every 100 iters:
  * `--eval_iters=100`
* stop training after 300 iters:
  * `--max_iters=300`

### Sample: view output from the saved model   
`!cd /content/nanoGPT && python sample.py --dtype=float32 --num_samples=5 --max_new_tokens=5 --start="to be"`

`sample.py` arguments explained:

* number of seperate examples output:
  * `--num_samples=5`
* ~ number of words per example to output (words ~ tokens x 0.75) 
  * `--max_new_tokens=5`
* start each output example with:
  * `--start="to be"`

**Full Colab Code:**
```
  # download repo
  !git clone https://github.com/karpathy/nanoGPT.git
  
  # install dependencies
  pip install tiktoken transformers
  
  # download shakespeare dataset into ./data/shakespeare
  !cd /content/nanoGPT/data/shakespeare/ && python prepare.py
  
  # finetune gpt-medium with "smaller Transformer" on GPU, model in ./out. (300 iters seems to have lowest val loss) 
  !cd /content/nanoGPT/ && python train.py --dataset=shakespeare --n_layer=4 --n_head=4 --n_embd=64 --compile=False --block_size=64 --batch_size=8 --init_from=gpt2-medium --dtype=float32 --eval_interval=100 --eval_iters=100 --max_iters=300
  
  # print 5 samples, with 5 tokens, starting with "to be"
  !cd /content/nanoGPT && python sample.py --dtype=float32 --num_samples=5 --max_new_tokens=5 --start="to be"
```
