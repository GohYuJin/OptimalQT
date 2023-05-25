import torch

# https://github.com/jxmorris12/categorical-vae

def gumbel_distribution_sample(shape: torch.Size, eps=1e-20) -> torch.Tensor:
    """Samples from the Gumbel distribution given a tensor shape and value of epsilon.
    
    note: the \eps here is just for numerical stability. The code is basically just doing
            > -log(-log(rand(shape)))
    where rand generates random numbers on U(0, 1). 
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_distribution_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Adds Gumbel noise to `logits` and applies softmax along the last dimension.
    
    Softmax is applied wrt a given temperature value. A higher temperature will make the softmax
    softer (less spiky). Lower temperature will make softmax more spiky and less soft. As
    temperature -> 0, this distribution approaches a categorical distribution.
    """
    assert len(logits.shape) == 2 # (should be of shape (b, n_classes))
    y = logits + gumbel_distribution_sample(logits.shape).to(logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits: torch.Tensor, temperature: float, batch=False) -> torch.Tensor:
    """
    Gumbel-softmax.
    input: [*, n_classes] (or [b, *, n_classes] for batch)
    return: flatten --> [*, n_class] a one-hot vector (or b, *, n_classes for batch)
    """
    input_shape = logits.shape
    if batch:
        assert len(logits.shape) == 3
        b, n, k = input_shape
        logits = logits.view(b*n, k)
    assert len(logits.shape) == 2
    y = gumbel_softmax_distribution_sample(logits, temperature)    
    n_classes = input_shape[-1] # TODO(jxm): check this!
    return y.view(input_shape)