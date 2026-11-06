import torch

from toxion.training.semi_supervised import (EMATeacher, SemiSupervisedConfig, consistency_loss, pseudo_targets,
                                             ramp_factor, unlabeled_loss)


def test_pseudo_targets_gate_by_confidence():
    t = pseudo_targets(torch.tensor([[0.95, 0.5, 0.02, 0.9, 0.05]]), hi=0.9, lo=0.05)
    assert t[0, 0] == 1 and t[0, 2] == 0 and t[0, 3] == 1 and t[0, 4] == 0 and torch.isnan(t[0, 1])


def test_unconfident_batch_gives_no_pseudo_label_loss():
    cfg = SemiSupervisedConfig(lambda_cons=0.0, lambda_pl=1.0)
    logits = torch.zeros(4, 12)
    assert unlabeled_loss(logits, torch.full((4, 12), 0.5), cfg, ramp=1.0).item() == 0.0


def test_consistency_is_zero_for_identical_views_and_positive_otherwise():
    logits = torch.randn(3, 12)
    assert consistency_loss(logits, torch.sigmoid(logits)).item() < 1e-12
    assert consistency_loss(logits, torch.sigmoid(logits + 3)).item() > 0


def test_ema_teacher_tracks_student_without_gradients():
    student = torch.nn.Linear(3, 2)
    teacher = EMATeacher(student, decay=0.5)
    assert all(not p.requires_grad for p in teacher.model.parameters())
    with torch.no_grad():
        student.weight.add_(1.0)
    before = teacher.model.weight.clone()
    teacher.update(student)
    assert torch.allclose(teacher.model.weight, before + 0.5)


def test_ramp():
    assert ramp_factor(0, 4) == 0.25 and ramp_factor(10, 4) == 1.0 and ramp_factor(0, 0) == 1.0
