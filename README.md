# nanoGPTshakespeare

* [Based on karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

Colab Code:
```
  !git clone https://github.com/karpathy/nanoGPT.git
  
  pip install tiktoken transformers
  
  !cd /content/nanoGPT/data/shakespeare/ && python prepare.py
  
  !cd /content/nanoGPT/ && python train.py --dataset=shakespeare --n_layer=4 --n_head=4 --n_embd=64 --compile=False --eval_iters=1 --block_size=64 --batch_size=8 --init_from=gpt2-medium --dtype=float32 --eval_interval=100
  
  !cd /content/nanoGPT && python sample.py --dtype=float32 --num_samples=5 --max_new_tokens=5 --start="to be"
```
