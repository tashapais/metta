from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext

# Keep: heavy module + manages circular dependency (loss <-> trainer)
if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class ActionSupervisedConfig(LossConfig):
    action_loss_coef: float = Field(default=1, ge=0)
    teacher_led_proportion: float = Field(default=0.0, ge=0, le=1.0)  # at 0.0, it's purely student-led

    # Controls whether to add the imitation loss to the environment rewards.
    add_action_loss_to_rewards: bool = Field(default=False)
    action_reward_coef: float = Field(default=0.01, ge=0)  # value is awild ass guess

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "ActionSupervised":
        """Create ActionSupervised loss instance."""
        return ActionSupervised(policy, trainer_cfg, vec_env, device, instance_name, self)


class ActionSupervised(Loss):
    __slots__ = ("rollout_batch_size", "teacher_mask")

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        cfg: "ActionSupervisedConfig",
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, cfg)

    def get_experience_spec(self) -> Composite:
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        action_spec = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int32)
        boolean = UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool)

        return Composite(
            actions=action_spec,
            teacher_actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long),
            teacher_mask=boolean,
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            if not hasattr(self, "rollout_batch_size") or self.rollout_batch_size != td.batch_size.numel():
                self._create_teacher_mask(td.batch_size.numel())

            self.policy.forward(td)

            if bool(self.teacher_mask.any()):
                teacher_actions = td["teacher_actions"].to(dtype=torch.long)
                td["actions"][self.teacher_mask] = teacher_actions.to(td["actions"].dtype)[self.teacher_mask]
                if "act_log_prob" in td.keys():
                    td["act_log_prob"][self.teacher_mask] = 0.0

            td["teacher_mask"] = self.teacher_mask

        env_slice = self._training_env_id(context)
        assert self.replay is not None
        self.replay.store(data_td=td, env_id=env_slice)

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"full_log_probs", "act_log_prob"} if self.cfg.add_action_loss_to_rewards else {"full_log_probs"}

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]
        policy_td = shared_loss_data["policy_td"]

        policy_full_log_probs = policy_td["full_log_probs"].reshape(minibatch.shape[0], minibatch.shape[1], -1)
        teacher_actions = minibatch["teacher_actions"]
        # get the student's logprob for the action that the teacher chose
        student_log_probs = policy_full_log_probs.gather(dim=-1, index=teacher_actions.unsqueeze(-1))
        student_log_probs = student_log_probs.reshape(minibatch.shape[0], minibatch.shape[1])

        loss = -student_log_probs.mean() * self.cfg.action_loss_coef

        self.loss_tracker["supervised_action_loss"].append(float(loss.item()))

        # --------------------------Add action loss to rewards as per Matt's doc----------------------------------
        if self.cfg.add_action_loss_to_rewards:
            minibatch["rewards"] = (
                minibatch["rewards"] + self.cfg.action_reward_coef * policy_td["act_log_prob"].detach()
            )
            # NOTE: we should somehow normalize the policy loss before adding it to rewards, perhaps exponentiate then
            # softplus?

        return loss, shared_loss_data, False

    def on_train_phase_end(self, context: ComponentContext | None = None) -> None:
        if hasattr(self, "rollout_batch_size"):
            self._create_teacher_mask(self.rollout_batch_size)
        super().on_train_phase_end(context)

    def _create_teacher_mask(self, batch_size: int) -> None:
        self.rollout_batch_size = int(batch_size)
        rand = torch.rand(self.rollout_batch_size, device=self.device)
        self.teacher_mask = rand < float(self.cfg.teacher_led_proportion)
