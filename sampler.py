
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn




# Sujay's sampler

# --- MODIFICATION START: Replace entire Sinkhorn implementation with a memory-safe alternative ---

class DifferentiableTopK(nn.Module):
    """A memory-efficient differentiable Top-K operator using a sigmoid approximation."""
    def __init__(self, k: int, temperature: float = 0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature
        logger.info(f"Initialized DifferentiableTopK with k={k} and temp={temperature}")

    def forward(self, logits: Tensor) -> Tensor:
        # This approach avoids creating the large n x k cost matrix.
        if self.k < 1:
            return torch.zeros_like(logits)

        # Find the k-th largest value in the logits, which will serve as our threshold.
        # This is efficient and doesn't require a full sort.
        if self.k < logits.size(-1):
            kth_value, _ = torch.kthvalue(logits, logits.size(-1) - self.k)
        else: # Handle case where k is equal to or greater than the number of logits
            kth_value = logits.min() - 1

        # Create a soft mask using a sigmoid function.
        # Values much larger than the threshold will be pushed towards 1.
        # Values much smaller than the threshold will be pushed towards 0.
        soft_mask = torch.sigmoid((logits - kth_value) / self.temperature)
        return soft_mask
    

def gumbel_top_k_sampling_v2(logits, k, temperature=1.0, eps=1e-10) -> Tensor:
    """
    Alternative implementation using continuous relaxation of top-k operation.
    This version maintains better gradients by avoiding hard masking.

    The code for this method is shared by Wuga, see
    https://claude.ai/public/artifacts/138b83ce-f40f-495f-81a7-bc8bd7416fce

    See Also
    [1] https://arxiv.org/pdf/1903.06059
    [2] https://papers.nips.cc/paper_files/paper/2014/file/937debc749f041eb5700df7211ac795c-Paper.pdf
    [3] https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html

    Args:
        logits (torch.Tensor): Input logits of shape (..., vocab_size)
        k (int): Number of top elements to sample
        temperature (float): Temperature parameter
        eps (float): Small constant for numerical stability

    Returns:
        torch.Tensor: Soft top-k samples
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    gumbel_logits = logits + gumbel_noise

    # Use continuous relaxation of top-k
    # Sort the gumbel logits to find the k-th largest value
    sorted_gumbel, _ = torch.sort(gumbel_logits, dim=-1, descending=True)
    threshold = sorted_gumbel[..., k - 1 : k]  # k-th largest value

    # Create soft mask using sigmoid
    soft_mask = torch.sigmoid((gumbel_logits - threshold) / temperature)

    # Apply soft mask and normalize
    masked_logits = logits * soft_mask
    return F.softmax(masked_logits / temperature, dim=-1)



class GumbelSampler(nn.Module):
    """Wrapper for the original Gumbel sampler to make the interface consistent."""
    def __init__(self, k: int, temperature: float = 1.0):
        super().__init__()
        self.k = k
        self.temperature = temperature
        logger.info(f"Initialized GumbelSampler with k={k} and temp={temperature}")

    def forward(self, logits: Tensor) -> Tensor:
        return gumbel_top_k_sampling_v2(logits, self.k, self.temperature)

# # --- MODIFICATION END ---

            
#         # --- MODIFICATION START: Instantiate the correct memory-safe sampler ---
#         if self.config.sampling_method == 'sinkhorn':
#             self.sampler = DifferentiableTopK(k=self.K).to(self.device)
#         elif self.config.sampling_method == 'gumbel':
#             self.sampler = GumbelSampler(k=self.K).to(self.device)
#         else:
#             raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")
#         # --- MODIFICATION END ---
    
    
                
#                 # --- MODIFICATION START: Call the sampler instance directly ---
#                 mask = self.sampler(emperical_Q_logits)
#                 # --- MODIFICATION END ---
                