"""Helper functions"""

import torch
from tensordict import TensorDict
from torch.distributions import Categorical
import torch.distributions as d


SAMPLE_LP = ("agents", "_tmp_action_log_prob") 
ACTION_P   = ("agents","_tmp_action")
FLAT = ("agents", "action")
ROOT_LP  = ("_prev_log_prob_struct",)  # this is loss_module.tensor_keys.sample_log_prob

def ensure_composite_action_on(td):
    # make sure ('agents','_tmp_action') is a TD with a1/a2
    act_parent = td.get(ACTION_P) if ACTION_P in td.keys(True) else None
    if not isinstance(act_parent, TensorDict):
        flat = td.get(env.action_key)                     # (..., n_agents, 2)
        a1 = flat[..., 0].to(torch.long)
        a2 = flat[..., 1].to(torch.long)
        agents_td = td.get(("agents",))
        bs = agents_td.batch_size
        names = getattr(agents_td, "names", None)
        parent = TensorDict({}, batch_size=bs, names=names) if names else TensorDict({}, batch_size=bs)
        parent.set("a1", a1)
        parent.set("a2", a2)
        td.set(ACTION_P, parent)
    return td



def ensure_tmp_and_logprob(td):
    flat = td.get(FLAT)
    
    a1 = flat[..., 0].to(torch.long)
    a2 = flat[..., 1].to(torch.long)
    td.set(ACTION_P + ("a1",), a1)
    td.set(ACTION_P + ("a2",), a2)

    # --- CREATE PARENT WITH ROOT BATCH SIZE & NAMES ---
    root_bs = td.batch_size                      # e.g. (time,)
    root_names = getattr(td, "names", None)      # ['time'] if named
    parent = td.get(SAMPLE_LP) if SAMPLE_LP in td.keys(True) else None
    if not isinstance(parent, TensorDict) or parent.batch_size != root_bs:
        if root_names is not None:
            td.set(SAMPLE_LP, TensorDict({}, batch_size=root_bs, names=root_names))
        else:
            td.set(SAMPLE_LP, TensorDict({}, batch_size=root_bs))

    # compute branch-wise log-probs
    a1_logits = td.get(("agents","a1_logits"))
    a2_logits = td.get(("agents","a2_logits"))
    lp1 = d.Categorical(logits=a1_logits).log_prob(a1)    # shape (time, n_agents)
    lp2 = d.Categorical(logits=a2_logits).log_prob(a2)

    td.set(SAMPLE_LP + ("a1",), lp1)
    td.set(SAMPLE_LP + ("a2",), lp2)

    return td



def rebuild_prev_logprob_root(td):
    # actions & logits
    a1 = td.get(ACTION_P + ("a1",))
    a2 = td.get(ACTION_P + ("a2",))
    a1_logits = td.get(("agents","a1_logits"))
    a2_logits = td.get(("agents","a2_logits"))

    # per-branch prev log-probs (shape: (time, n_agents))
    lp1 = d.Categorical(logits=a1_logits).log_prob(a1)
    lp2 = d.Categorical(logits=a2_logits).log_prob(a2)

    # === IMPORTANT: use AGENTS-level batch size for the ROOT parent ===
    agents_td    = td.get(("agents",))           # TD with batch_size == (time, n_agents)
    bs_agents    = agents_td.batch_size          # e.g., torch.Size([2, 10])

    # Nuke any previous value (guard against Tensor vs TD collisions)
    if ROOT_LP in td.keys(True):
        try:
            td.del_(ROOT_LP)
        except Exception:
            td = td.exclude(ROOT_LP)

    # Build a ROOT-level TD with batch_size == (time, n_agents)
    root_td   = TensorDict({}, batch_size=bs_agents)
    agents_lp = TensorDict({}, batch_size=bs_agents)
    act_lp    = TensorDict({}, batch_size=bs_agents)

    act_lp.set("a1_log_prob", lp1)
    act_lp.set("a2_log_prob", lp2)
    agents_lp.set("_tmp_action", act_lp)
    root_td.set("agents", agents_lp)

    td.set(ROOT_LP, root_td)
    
    return td