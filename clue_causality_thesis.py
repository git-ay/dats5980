import torch.nn.functional as F
from torch import nn
import torch
import spacy


def is_token_clue(doc:list[str]) -> list[bool]:
    
    '''
    Args:
    
        docs : list[str]
            A sentence containing words (in str).
        
    Returns:
    
        is_clue : list[bool]
            A list containing True/False corresponding to the words. 
            True if a word is a clue.
            False if a word is not a clue. 
        
    '''
    
    ENG_CLUES_1 = set([
        'because', 'after', 'as', 'from', 'for', 
        'since', 'with', 'consequently', 'therefore', 
        'accordingly', 'thus', 'so', 'hence', 'affected', 'will'
    ]) # Should we add `will *` for all *?
    ENG_CLUES_2 = set([
        'because of', 'due to', 'for the reason that', 'meant that', 'prompted by', 'helped by', 'fuelled by', 'partly on', 'owing to'
    ])
    ENG_CLUES_3 = set([
        'as a result', 'as a consequence', 'was aided by', 'for this reason'
    ])

    doc = [ token.lower() for token in doc ]

    is_clue = [ False ] * len(doc)
    for i, token in enumerate(doc):

        if token in ENG_CLUES_1:
            is_clue[ i ] = True

        if i >= 1:
            prev1_token = doc[ i-1 ]
            if f'{prev1_token} {token}' in ENG_CLUES_2:
                is_clue[ i-1 ] = is_clue[ i ] = True

        if i >= 2:
            prev2_token = doc[ i-2 ]
            if f'{prev2_token} {prev1_token} {token}' in ENG_CLUES_3:
                is_clue[ i-2 ] = is_clue[ i-1 ] = is_clue[ i ] = True

    return is_clue


class ClueCausalityExtractionThesis(nn.Module):
    
    ENG_CLUES_1 = set([
        'because', 'after', 'as', 'from', 'for', 
        'since', 'with', 'consequently', 'therefore', 
        'accordingly', 'thus', 'so', 'hence', 'affected', 'will'
    ]) # Should we add `will *` for all *?
    ENG_CLUES_2 = set([
        'because of', 'due to', 'for the reason that', 'meant that', 'prompted by', 'helped by', 'fuelled by', 'partly on', 'owing to'
    ])
    ENG_CLUES_3 = set([
        'as a result', 'as a consequence', 'was aided by', 'for this reason'
    ])
    
    def __init__(
            self,
            bert_embed_func,
            token_to_embed: dict = {}
        ) -> None:
        super(ClueCausalityExtractionThesis, self).__init__()
        
        dummy_embed = bert_embed_func(['INFER EMBEDDING SIZE']*2)
        assert all([ dummy_embed.shape[0]==2, dummy_embed.shape[-1] > 0 ]), 'bert_embed_func is not working properly ... please check!'
        
        self.bert_embed_func = bert_embed_func # This function should return a tensor.FloatTensor of dim (num batch, embedding size).
        bert_embed_len = dummy_embed.shape[-1] # embedding size.
        self.token_to_embed = token_to_embed # initialized with {} if not provided.
        
        self.W_g = nn.Linear(in_features=bert_embed_len, out_features=bert_embed_len, bias=True)
        self.alpha_left = nn.Parameter(data=torch.randn(bert_embed_len), requires_grad=True)
        self.alpha_right = nn.Parameter(data=torch.randn(bert_embed_len), requires_grad=True)
        
        HALF_SIZE = int(bert_embed_len/2)
        self.gru_hidden = nn.Parameter(data=torch.zeros(1, HALF_SIZE, requires_grad=False), requires_grad=False)
        self.gru = nn.GRU(input_size=bert_embed_len, hidden_size=HALF_SIZE, num_layers=1, bidirectional=False, batch_first=True)
        
        self.W_cause = nn.Linear(in_features=bert_embed_len + HALF_SIZE, out_features=3, bias=True)
        self.W_effect = nn.Linear(in_features=bert_embed_len + HALF_SIZE, out_features=3, bias=True)
                
        if '[CLS]' not in self.token_to_embed:
            self.store_additional_word_embeddings(['[CLS]'])
        
    def forward(
            self,
            docs:list[spacy.tokens.doc.Doc],
            is_clues:list[list[bool]]
        ):
        '''
        docs: A list of spacy.tokens.doc.Doc containing texts that has Causes and Effects.
              e.g. [ "Strength of GraphAttention!",
                     "BIO classification will change the world!" ]
                   where, "Stength of GraphAttention!" is spacy.tokens.doc.Doc
                          `of` and `will` are the clues. 
                   
        is_clues: A list of bool containing a batch(=lists) of booleans indicating.
                  (True=This token is the part of clue expression, False=This token is not the part of clue expression)
                  
                  Using the example sentences in docs:,
                  e.g. [ [False, True, False, False], # Assume that ["Strength", "of", "GraphAttention", "!"] are tokens. `of` is the only clue. 
                         [False, False, True, False, False, False, False] ]
        '''
        self.bert_to_embed(docs)
        
        G_W, C = [], []
        for doc, is_clue in zip(docs, is_clues):

            G_W.append([])
            C.append([])

            cls_embed = self.token_to_embed['[CLS]']
            G_W[-1].append( cls_embed )
            C[-1].append( cls_embed )
            
            clue_pos = 0
            for i, (token_i, ic) in enumerate(zip(doc, is_clue), start=1):

                t_i = self.token_to_embed[ token_i.text ]
                new_t_i = self.graph_attention(token_i, t_i) if len(tuple(token_i.children)) > 0 else t_i # new_t_i is g_i in the thesis. 
                
                G_W[-1].append( new_t_i )
                
                if ic:
                    C[-1].append( new_t_i )
            
            G_W[-1] = torch.vstack(G_W[-1])
            C[-1] = torch.vstack(C[-1])
        
        h_clues = self.h_clues(C)
        O_causes, O_effects = self.mlp_cause_and_effect(G_W, h_clues)
        
        return O_causes, O_effects
        
    def bert_to_embed(
            self,
            docs:list[spacy.tokens.doc.Doc]
        ) -> None:
        '''
        docs: A list of spacy.tokens.doc.Doc containing texts that has Causes and Effects.
              e.g. [ "Strength of GraphAttention!",
                     "BIO classification will change the world!" ]
                   where, "I like GraphAttention!" is spacy.tokens.doc.Doc        
        '''            
        for doc in docs:
            self.store_additional_word_embeddings([ d.text for d in doc ])    
    
    def store_additional_word_embeddings(
            self,
            tokens:list[str]
        ) -> None:
        
        xlm_reqs = set() # Get unique words in the tokens that are not in self.token_to_embed
        for token in tokens:
            if token not in self.token_to_embed:
                xlm_reqs.add(token)
        xlm_reqs = list(xlm_reqs)

        if len(xlm_reqs) > 0: # If there are unique words, 
            for xr, embed in zip( xlm_reqs, self.bert_embed_func(xlm_reqs) ): # self.bert_embed_func should give back the output to the right device. 
                self.token_to_embed[ xr ] = embed # map the unique words to the embedding from bert. 
            
    def graph_attention(
            self,
            token_i:spacy.tokens.token.Token,
            t_i:torch.FloatTensor
        ):
        '''
        Please refer to the thesis https://link.springer.com/article/10.1007/s00354-023-00233-2. 
        This is the implementation of GAT.
        Returns t_i' in the thesis for each token w_i.
        '''
        aT_Wg_ti = torch.dot( # aT_Wg_ti = (num root = 1,)
            self.alpha_left, 
            self.W_g( t_i )
        )
        Wg_tks = self.W_g( # Wg_tks = (num children, bert_embed_len)
            torch.vstack([
                self.token_to_embed[ token_k.text ] for token_k in token_i.children 
            ])
        )
        a_ij = aT_Wg_ti + torch.mv( input=Wg_tks, vec=self.alpha_right ) # a_ij = (num children = |N_i| in the thesis,)
        a_ij = F.softmax(
            F.leaky_relu( a_ij, negative_slope=0.2 ),
            dim=0
        )
        new_t_i = t_i + a_ij @ Wg_tks # new_t_i := t_i' in the thesis.

        return new_t_i
    
    def h_clues(
            self, 
            C:list[torch.FloatTensor]
        ):
        return [ self.gru(c, self.gru_hidden)[1].reshape(1,-1) for c in C ] # [1] returns the last layer of self.gru[1] = h_clue = GRU(c) in the thesis. 
    
    def mlp_cause_and_effect(
            self,
            G_W:list[torch.FloatTensor],
            h_clues:list[torch.FloatTensor]
        ):
        
        O_causes, O_effects = [], []
        for gw, h_clue in zip(G_W, h_clues):
            
            gwi_h_clue = torch.hstack([
                gw,
                h_clue[ [0] * gw.shape[0] ]
            ])
            
            O_causes.append(
                self.W_cause( gwi_h_clue )[1:]
            )
            O_effects.append(
                self.W_effect( gwi_h_clue )[1:]
            )
            
        return O_causes, O_effects