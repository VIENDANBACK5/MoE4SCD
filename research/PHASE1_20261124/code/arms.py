"""Arms of Phase 1 · Prototype 1 (../PROTOCOL_M2_PROTOTYPE.md).

Inputs are frozen RoI features F [B,256,R,R] and the candidate mask M [B,1,R,R].
- B1: decision only (the partition is a non-learned GMM at inference).
- B2: decision + two independent sigmoid mask heads.
- M2: parent-conditioned two-hypothesis competition, with joint hypothesis/membership updates (slot-attention style)
  and a decision that reads the outcome of the competition.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

D = 128


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(conv_block(257, D), conv_block(D, D), nn.Conv2d(D, D, 3, padding=1))

    def forward(self, F, M):
        return self.net(torch.cat([F, M], 1))


def pooled(X, M):
    w = M.flatten(1); x = X.flatten(2)
    mean = (x * w[:, None]).sum(-1) / w.sum(-1, keepdim=True).clamp_min(1e-6)
    mx = (x * w[:, None]).amax(-1)
    return torch.cat([mean, mx], 1)                                    # [B, 2D]


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o))


class B1(nn.Module):
    def __init__(self):
        super().__init__(); self.adapter = Adapter(); self.dec = mlp(2 * D, 256, 1)

    def forward(self, F, M):
        X = self.adapter(F, M); return {"logit": self.dec(pooled(X, M))[:, 0]}


class B2(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter(); self.dec = mlp(2 * D, 256, 1)
        self.head = nn.Sequential(conv_block(D, D), conv_block(D, D), nn.Conv2d(D, 2, 1))   # extra block for parameter parity

    def forward(self, F, M):
        X = self.adapter(F, M)
        return {"logit": self.dec(pooled(X, M))[:, 0], "mask_logits": self.head(X)}         # independent sigmoids


class M2(nn.Module):
    """Two object hypotheses + one rest-of-scene hypothesis compete for every RoI pixel (softmax over hypotheses).
    Hypotheses start from the parent's principal split and are updated jointly with the memberships, T times."""

    def __init__(self, T=3, power_iters=6):
        super().__init__()
        self.adapter = Adapter(); self.T, self.P = T, power_iters
        self.v0 = nn.Parameter(torch.randn(D) / math.sqrt(D)); self.alpha = nn.Parameter(torch.ones(1))
        self.ln_in, self.ln_s, self.ln_m = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.Wq, self.Wk, self.Wv = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.gru = nn.GRUCell(D, D); self.mlp = mlp(D, 2 * D, D)
        self.dec = mlp(2 * D + 2 * D + 3, 256, 1)

    def init_slots(self, x, w):
        """x [B,N,D] features, w [B,N] parent weights -> slots [B,3,D] (rest, obj1, obj2)."""
        ws = w.sum(1, keepdim=True).clamp_min(1e-6)
        mu = (x * w[..., None]).sum(1) / ws
        xc = x - mu[:, None]
        v = self.v0.expand(x.shape[0], -1)
        for _ in range(self.P):                                          # power iteration on the parent covariance
            cv = ((xc @ v[..., None])[..., 0] * w)[..., None] * xc        # [B,N,D]
            v = Fn.normalize(cv.sum(1) / ws, dim=-1)
        lam = (((xc @ v[..., None])[..., 0] ** 2) * w).sum(1, keepdim=True) / ws
        s = self.alpha * lam.clamp_min(1e-8).sqrt() * v
        wo = (1 - w); rest = (x * wo[..., None]).sum(1) / wo.sum(1, keepdim=True).clamp_min(1e-6)
        return torch.stack([rest, mu + s, mu - s], 1)

    def attend(self, slots, k):
        q = self.Wq(self.ln_s(slots))                                     # [B,3,D]
        return torch.einsum("bjd,bnd->bjn", q, k) / math.sqrt(D)        # logits [B,3,N]

    def forward(self, F, M):
        X = self.adapter(F, M); B, _, H, W = X.shape
        x = X.flatten(2).transpose(1, 2); w = M.flatten(1)
        xn = self.ln_in(x); k, v = self.Wk(xn), self.Wv(xn)
        slots = self.init_slots(x, w)
        for _ in range(self.T):
            a = self.attend(slots, k).softmax(1)                          # competition over hypotheses
            abar = a / a.sum(2, keepdim=True).clamp_min(1e-6)
            upd = torch.einsum("bjn,bnd->bjd", abar, v)
            slots = self.gru(upd.reshape(-1, D), slots.reshape(-1, D)).reshape(B, 3, D)
            slots = slots + self.mlp(self.ln_m(slots))
        logits = self.attend(slots, k)                                    # final memberships
        a = logits.softmax(1); ws = w.sum(1).clamp_min(1e-6)
        mass = (a[:, 1:] * w[:, None]).sum(2) / ws[:, None]              # [B,2] mass won inside the parent
        ent = (-(a.clamp_min(1e-8).log() * a).sum(1) * w).sum(1) / ws
        z1, z2 = slots[:, 1], slots[:, 2]
        feats = torch.cat([pooled(X, M), z1 + z2, (z1 - z2).abs(), mass.min(1, keepdim=True)[0],
                           mass.max(1, keepdim=True)[0], ent[:, None]], 1)   # symmetric in (z1, z2)
        return {"logit": self.dec(feats)[:, 0], "member_logits": logits.reshape(B, 3, H, W)}


ARMS = {"B1": B1, "B2": B2, "M2": M2}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------- losses ----------------
def soft_dice(p, t, valid):
    """p, t [B,H,W] in [0,1]; valid [B,H,W]. Returns per-sample 1 − Dice."""
    p, t = p * valid, t * valid
    return 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)


def best_perm(p1, p2, t, valid):
    """True where swapping (t1, t2) gives the lower Dice cost."""
    same = soft_dice(p1, t[:, 0], valid) + soft_dice(p2, t[:, 1], valid)
    swap = soft_dice(p1, t[:, 1], valid) + soft_dice(p2, t[:, 0], valid)
    return swap < same


def mask_loss_B2(out, t, ign):
    valid = (1 - ign).float(); ml = out["mask_logits"]; p = ml.sigmoid()
    sw = best_perm(p[:, 0], p[:, 1], t, valid)
    tt = torch.where(sw[:, None, None, None], t.flip(1), t)
    bce = (Fn.binary_cross_entropy_with_logits(ml, tt, reduction="none") * valid[:, None]).sum((1, 2, 3)) / (2 * valid.sum((1, 2)).clamp_min(1))
    dice = soft_dice(p[:, 0], tt[:, 0], valid) + soft_dice(p[:, 1], tt[:, 1], valid)
    return (bce + dice).mean()


def mask_loss_M2(out, t, ign):
    valid = (1 - ign).float(); lg = out["member_logits"]; a = lg.softmax(1)
    sw = best_perm(a[:, 1], a[:, 2], t, valid)
    tt = torch.where(sw[:, None, None, None], t.flip(1), t)
    tgt = torch.cat([(1 - tt.sum(1, keepdim=True)).clamp_min(0), tt], 1)
    tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-6)
    ce = (-(tgt * lg.log_softmax(1)).sum(1) * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    dice = soft_dice(a[:, 1], tt[:, 0], valid) + soft_dice(a[:, 2], tt[:, 1], valid)
    return (ce + dice).mean()


MASK_LOSS = {"B1": None, "B2": mask_loss_B2, "M2": mask_loss_M2}
