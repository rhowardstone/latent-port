import torch
from torch import nn

from visual_encoder.checkpoints import checkpoint, clear, resume


def test_save_resume_round_trip(tmp_path):
    path = tmp_path / "ck.pt"
    m = nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # take a step so optimizer has state
    m(torch.randn(2, 4)).sum().backward()
    opt.step()
    checkpoint(path, step=123, model=m, opt=opt)

    m2 = nn.Linear(4, 4)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    step = resume(path, "cpu", model=m2, opt=opt2)
    assert step == 124  # resumes at step+1
    assert torch.allclose(m.weight, m2.weight)


def test_resume_missing_returns_zero(tmp_path):
    assert resume(tmp_path / "nope.pt", "cpu", model=nn.Linear(2, 2)) == 0


def test_clear_removes(tmp_path):
    path = tmp_path / "ck.pt"
    checkpoint(path, 5, model=nn.Linear(2, 2))
    assert path.exists()
    clear(path)
    assert not path.exists()
