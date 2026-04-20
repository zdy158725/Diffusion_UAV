import torch
import torch.nn as nn


class StructuredUAVObsEncoder(nn.Module):
    def __init__(
        self,
        max_agents: int = 6,
        agent_state_dim: int = 7,
        team_dim: int = 2,
        slot_count: int = 6,
        role_count: int = 4,
        hidden_dim: int = 256,
        relation_hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_agents = int(max_agents)
        self.agent_state_dim = int(agent_state_dim)
        self.hidden_dim = int(hidden_dim)

        self.agent_state_encoder = nn.Sequential(
            nn.Linear(agent_state_dim, relation_hidden_dim),
            nn.Mish(),
            nn.Linear(relation_hidden_dim, hidden_dim),
        )
        self.team_encoder = nn.Linear(team_dim, hidden_dim)
        self.slot_embedding = nn.Embedding(slot_count, hidden_dim)
        self.role_embedding = nn.Embedding(role_count, hidden_dim)
        self.relation_encoder = nn.Sequential(
            nn.Linear(8, relation_hidden_dim),
            nn.Mish(),
            nn.Linear(relation_hidden_dim, hidden_dim),
        )
        self.self_query = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        agent_obs: torch.Tensor,
        agent_team: torch.Tensor,
        agent_valid: torch.Tensor,
        agent_social_mask: torch.Tensor,
        agent_role_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_obs: [B, T_obs, A, 7]
        agent_team: [B, A, 2]
        agent_valid: [B, A]
        agent_social_mask: [B, A, A]
        agent_role_id: [B, A]
        return: [B, T_obs, hidden_dim]
        """
        if agent_obs.ndim != 4:
            raise ValueError(f"Expected agent_obs ndim=4, got {agent_obs.shape}")

        batch_size, n_obs_steps, max_agents, _ = agent_obs.shape
        if max_agents != self.max_agents:
            raise ValueError(
                f"Expected max_agents={self.max_agents}, got agent_obs shape {agent_obs.shape}"
            )

        device = agent_obs.device
        dtype = agent_obs.dtype

        slot_ids = torch.arange(self.max_agents, device=device, dtype=torch.long)
        slot_embed = self.slot_embedding(slot_ids).view(1, 1, self.max_agents, self.hidden_dim)
        role_embed = self.role_embedding(agent_role_id.long()).unsqueeze(1)
        team_embed = self.team_encoder(agent_team.to(dtype=dtype)).unsqueeze(1)
        agent_embed = self.agent_state_encoder(agent_obs) + slot_embed + role_embed + team_embed

        self_state = agent_obs[:, :, 0:1, :]
        rel_pos = agent_obs[:, :, :, 0:3] - self_state[:, :, :, 0:3]
        rel_vel = agent_obs[:, :, :, 3:6] - self_state[:, :, :, 3:6]
        hp_diff = agent_obs[:, :, :, 6:7] - self_state[:, :, :, 6:7]
        self_team = agent_team[:, 0:1, :].unsqueeze(1).to(dtype=dtype)
        same_team = (
            agent_team.unsqueeze(1).to(dtype=dtype) * self_team
        ).sum(dim=-1, keepdim=True)
        same_team = same_team.expand(-1, n_obs_steps, -1, -1)
        relation_feat = torch.cat([rel_pos, rel_vel, hp_diff, same_team], dim=-1)
        relation_embed = self.relation_encoder(relation_feat)
        memory = agent_embed + relation_embed

        flat_memory = memory.reshape(batch_size * n_obs_steps, self.max_agents, self.hidden_dim)
        flat_query = self.self_query(memory[:, :, 0:1, :]).reshape(
            batch_size * n_obs_steps, 1, self.hidden_dim
        )
        flat_key = self.key_proj(flat_memory)
        flat_value = self.value_proj(flat_memory)

        valid_mask = agent_valid > 0.5
        target_mask = agent_social_mask[:, 0, :] > 0.5
        attn_mask = (valid_mask & target_mask).unsqueeze(1).expand(-1, n_obs_steps, -1)
        key_padding_mask = ~attn_mask.reshape(batch_size * n_obs_steps, self.max_agents)
        key_padding_mask[:, 0] = False

        attn_out, _ = self.attn(
            query=flat_query,
            key=flat_key,
            value=flat_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        scene_token = self.output_proj((flat_query + attn_out).squeeze(1))
        return scene_token.reshape(batch_size, n_obs_steps, self.hidden_dim)
