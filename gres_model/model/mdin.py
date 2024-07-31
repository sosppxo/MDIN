import torch
import torch.nn as nn
from .sample_model import SamplingModule
from torch_scatter import scatter_max, scatter_mean, scatter
import torch.nn.functional as F
# modified torch multihead attention
from ..torch.nn import MultiheadAttention
# graph
from .graph.graph_transformer_net import GraphTransformerNet
from .graph.layers.graph_transformer_edge_layer import GraphTransformerLayer, GraphTransformerSubLayer

# BRIEF predict object position, size and class.
class ThreeLayerMLP(nn.Module):
    """A 3-layer MLP with normalization and dropout."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            # nn.LayerNorm(dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, dim, 1, bias=False),
            # nn.LayerNorm(dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x):
        """Forward pass, x can be (B, dim, N)."""
        return self.net(x)

class DDI(nn.Module):

    def __init__(
            self,
            hidden_dim,
            out_dim,
            n_heads,
            dropout=0.0,
            layer_norm=True, 
            batch_norm=False, 
            residual=True, 
            use_bias=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.graph_attn = GraphTransformerSubLayer(hidden_dim, out_dim, n_heads, dropout, layer_norm, batch_norm)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def graph2batch(self, batched_graph):
        node_num = batched_graph.batch_num_nodes()
        batch_offsets = torch.cat([torch.tensor((0,), dtype=torch.int).to(batched_graph.device), node_num.cumsum(0).int()], dim=0)
        batch_data, batch_masks = self.get_batches(batched_graph.ndata['h'], batch_offsets)
        return batch_data, batch_masks
    
    def batch2graph(self, batch_data, batch_masks):
        B = batch_data.shape[0]
        batch_x = []
        for i in range(B):
            batch_x.append(batch_data[i, (~batch_masks[i])])
        batch_x = torch.cat(batch_x, dim=0)
        return batch_x
    
    def forward(self, x, x_mask, batch_g, batch_x, batch_e, pe=None, cat='parallel'):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        # parallel
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        sa_output, _ = self.self_attn(q, k, x, key_padding_mask=x_mask)

        # graph-attention
        batch_x, batch_e = self.graph_attn(batch_g, batch_x, batch_e)
        batch_g.ndata['h'] = batch_x
        batch_g.edata['e'] = batch_e

        # transform batched graph to batched tensor
        ga_output, _ = self.graph2batch(batch_g)
        ga_output = torch.cat([ga_output, torch.zeros(B, x_mask.shape[1]-ga_output.shape[1], ga_output.shape[-1]).to(ga_output.device)], dim=1)

        # residual connection
        output = self.dropout(sa_output + ga_output) + x
        output = self.norm(output)

        return output, batch_e

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_mask, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        B = query.shape[0]
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, query.shape[1], k.shape[1])
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output, output_weight, src_weight # (b, n_q, d_model), (b, n_q, n_v)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output
    
class MDIN(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        num_class=256,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        sampling_module=None,
        sampling_module_kv=None,
        kernel='top1',
        global_feat='mean',
        lang_att=False,
        contrastive_align_loss=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_class = num_class
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        
        self.lang_att = lang_att
        self.contrastive_align_loss = contrastive_align_loss

        H = 768
        self.lang_proj = nn.Linear(H, d_model)
        self.lang_norm = nn.LayerNorm(d_model)
        
        if sampling_module is not None:
            self.sampling_module = SamplingModule(**sampling_module)
        else:
            self.sampling_module = None

        if sampling_module_kv is not None:
            self.sampling_module_kv = SamplingModule(**sampling_module_kv)
        else:
            self.sampling_module_kv = None
       
        # TSQ
        self.query_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
       
        # MDIN
        self.qsa_layers = nn.ModuleList([])
        self.qqa_layers = nn.ModuleList([])
        self.qla_layers = nn.ModuleList([])
        self.qla_ffn_layers = nn.ModuleList([])
        if self.lang_att:
            self.lla_layers = nn.ModuleList([])
            self.lsa_layers = nn.ModuleList([])
            self.lsa_ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.qsa_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.qqa_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.qla_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.qla_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            if self.lang_att:
                self.lla_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
                self.lsa_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
                self.lsa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))

        # 0-th
        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        
        self.indi_embedding = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2), nn.Linear(2, 2))
        self.indi_norm = nn.LayerNorm(d_model)

        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.kernel = kernel
        self.global_feat = global_feat

        # if self.kernel == 'w_sum':
        #     self.query_weight_embedding = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2), nn.Linear(2, 2))
        #     self.w_norm = nn.LayerNorm(d_model)
        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_vision = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )
            self.contrastive_align_projection_text = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )

            
    
    def get_batches(self, x, batch_offsets):
        '''
        example:
            3 + 4 + 5 = 12
            batch_offsets = [0, 3, 7, 12]
            x = [12, D]
            ===> new_feats = [3(bs), 5, D]
            mask [3(bs), 5]
        '''
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)

            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def get_mask(self, query, mask_feats, batch_mask):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None
        return pred_masks, attn_masks

    def avg_lang_feat(self, lang_feats, lang_masks):
        lang_len = lang_masks.sum(-1)
        lang_len = lang_len.unsqueeze(-1)
        lang_len[torch.where(lang_len == 0)] = 1
        return (lang_feats * ~lang_masks.unsqueeze(-1).expand_as(lang_feats)).sum(1) / lang_len
    
    def avg_sp_l2_feat(self, query, batch_mask_l2):
        len = batch_mask_l2.sum(-1)
        len = len.unsqueeze(-1)
        len[torch.where(len == 0)] = 1
        return (query * ~batch_mask_l2.unsqueeze(-1)).sum(1) / len

    def prediction_head(self, query, mask_feats, batch_mask):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)
        return pred_scores, pred_masks, attn_masks

    
    def forward_iter_pred(self, x, fps_seed_sp, batch_offsets, lang_feats=None, lang_masks=None):
        """
        x [B*M, inchannel]
        """

        # lang_query
        lang_feats = self.lang_proj(lang_feats) 
        lang_feats = self.lang_norm(lang_feats)
        lang_masks = ~(lang_masks.bool())
        lang_query = lang_feats

        # instance feats
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)
        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)
        
        prediction_masks = []
        prediction_scores = []
        prediction_indis = []
        B = len(batch_offsets) - 1
        
        sample_inds = None
        ref_scores = None
        sample_inds_kv = None
        ref_scores_kv = None

        seed_sp = inst_feats.gather(dim=1, index=fps_seed_sp.long().unsqueeze(-1).repeat(1, 1, inst_feats.size(-1)))
         # sampling
        if hasattr(self, 'sampling_module') and self.sampling_module is not None:
            sample_inds, ref_scores = self.sampling_module(seed_sp, lang_query, None, lang_masks)
            sample_inds = sample_inds.long()
            # [B, N_q, D]
            sampled_seed = seed_sp.gather(dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, seed_sp.size(-1)))

            query = self.query_generator(sampled_seed)
        else:
            query = self.query_generator(seed_sp)

        proj_queries = []
        if self.contrastive_align_loss:
            proj_queries.append(
                F.normalize(
                    self.contrastive_align_projection_vision(query), p=2, dim=-1
                )
            )
        else:
            proj_queries.append(None)

        # 0-th prediction
        if self.kernel=='w_sum':
            pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
            
            pred_indis = self.indi_embedding(query)

            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
            prediction_indis.append(pred_indis)
        # else:
        #     pred_scores, pred_masks, attn_masks = self.prediction_head(self.avg_lang_feat(query, lang_masks).unsqueeze(1), mask_feats, batch_mask)
        #     _, _, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
        #     prediction_scores.append(pred_scores)
        #     prediction_masks.append(pred_masks)

        # sampling
        if hasattr(self, 'sampling_module_kv') and self.sampling_module_kv is not None:
            sample_inds_kv, ref_scores_kv = self.sampling_module_kv(inst_feats, lang_query, batch_mask, lang_masks)
            sample_inds_kv = sample_inds_kv.long()
            inst_feats = torch.gather(inst_feats, dim=1, index=sample_inds_kv.unsqueeze(-1).repeat(1,1,inst_feats.shape[-1]))
            batch_mask_sampled = torch.gather(batch_mask, dim=1, index=sample_inds_kv)
            attn_masks = torch.gather(attn_masks, dim=2, index=sample_inds_kv.unsqueeze(1).repeat(1,attn_masks.shape[1],1))


        # multi-round
        for i in range(self.num_layer):

            if self.lang_att:
                lang_query = self.lla_layers[i](lang_query, lang_masks)
                if hasattr(self, 'sampling_module_kv') and self.sampling_module_kv is not None:
                    lang_query, _, _ = self.lsa_layers[i](inst_feats, lang_query, batch_mask_sampled, None)
                else:
                    lang_query, _, _ = self.lsa_layers[i](inst_feats, lang_query, batch_mask, None)
                lang_query = self.lsa_ffn_layers[i](lang_query)

            if hasattr(self, 'sampling_module_kv') and self.sampling_module_kv is not None:
                query, _, src_weight = self.qsa_layers[i](inst_feats, query, batch_mask_sampled, None)
            else:
                query, _, src_weight = self.qsa_layers[i](inst_feats, query, batch_mask, attn_masks)

            query_rra = self.qqa_layers[i](query)
            query_rla, _, src_weight_1 = self.qla_layers[i](lang_query, query, lang_masks, None)
            
            if self.lang_att:
                lang_query = self.lla_layers[i](lang_query, lang_masks)
                lang_query, _, _ = self.lsa_layers[i](query, lang_query, None, None)
                lang_query = self.lsa_ffn_layers[i](lang_query)

            query = query + query_rla + query_rra 
            
            query = self.qla_ffn_layers[i](query)


            if self.kernel=='w_sum':
                pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
                pred_indis = self.indi_embedding(query)
                
            else:
                raise NotImplementedError
                
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
            prediction_indis.append(pred_indis)
            if self.contrastive_align_loss:
                proj_queries.append(
                    F.normalize(
                        self.contrastive_align_projection_vision(query), p=2, dim=-1
                    )
                )
            else:
                proj_queries.append(None)

            if hasattr(self, 'sampling_module_kv') and self.sampling_module_kv is not None:
                attn_masks = torch.gather(attn_masks, dim=2, index=sample_inds_kv.unsqueeze(1).repeat(1,attn_masks.shape[1],1))
              
        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(lang_query), p=2, dim=-1
            )
        else:
            proj_tokens = None

        return {
            'masks':
            pred_masks,
            'batch_mask':
            batch_mask,
            'scores':
            pred_scores,
            'indis':
            pred_indis, # [B, B_q, 2]
            'proj_queries':
            proj_queries[-1],
            'proj_tokens':
            proj_tokens,
            'sample_inds':
            sample_inds, # [B, K]
            'ref_scores':
            ref_scores, # [B, M]
            'ref_scores_kv':
            ref_scores_kv, # [B, M]
            'aux_outputs': [{
                'masks': a,
                'scores': b,
                'proj_queries': c,
                'indis': d,
            } for a, b, c, d in zip(
                prediction_masks[:-1],
                prediction_scores[:-1],
                proj_queries[:-1],
                prediction_indis[:-1],
            )],
        }

    def forward(self, x, fps_seed_sp, batch_offsets, lang_feats=None, lang_masks=None):
        if self.iter_pred:
            return self.forward_iter_pred(x, fps_seed_sp, batch_offsets, lang_feats, lang_masks)
        else:
            raise NotImplementedError
