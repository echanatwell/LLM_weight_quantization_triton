import time
from tqdm import tqdm
import torch

def measure_ppl(samples, model, tokenizer):
    device = model.device
    ppl = 0.
    start_time = time.time()
    for prompt in tqdm(samples):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        targets = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=targets)
            nll = outputs.loss
        ppl += torch.exp(nll)
    end_time = time.time()

    mean_ppl = ppl / len(samples)
    mean_time = (end_time - start_time) / len(samples)
    print()
    print(f'Perplexity: {mean_ppl:.4f}')
    print(f'Mean time per sample: {mean_time:.3f} s')
    return mean_ppl.item(), mean_time