import spacy
import polars as pl
from tqdm import tqdm
nlp = spacy.load("en_core_web_trf") 



def ce_char_start_end_pos(tce_pdf:pl.DataFrame):
    
    '''
    tce_pdf:
    
        Text                             	Cause	                            Effect
        str                              	str	                                str
        "It found that total U.S. healt…	"Part of the reason is that Med…	"It found that total U.S. healt…
        "Transat loss more than doubles…	"it works to complete Air Canad…	"Transat loss more than doubles"    
    
    
    returns:
    
        Text	Cause	Effect	cause_start	cause_end	effect_start	effect_end
        str	str	str	i64	i64	i64	i64
        "It found that total U.S. healt…	"Part of the reason is that Med…	"It found that total U.S. healt…	163	321	0	162
        "Transat loss more than doubles…	"it works to complete Air Canad…	"Transat loss more than doubles"	34	70	0	30 
    
    '''
    
    cause_starts, cause_ends = [], []
    effect_starts, effect_ends = [], []
    for i, (t, c, e) in enumerate(tce_pdf.iter_rows()):

        if isinstance(c, str):        
            c_start = t.find(c)
            c_end = c_start + len(c)
            cause_starts.append( c_start )
            cause_ends.append( c_end )
        else:
            cause_starts.append( 0 )
            cause_ends.append( 0 )

        if isinstance(e, str):
            e_start = t.find(e)
            e_end = e_start + len(e)
            effect_starts.append( e_start )
            effect_ends.append( e_end )
        else:
            effect_starts.append( 0 )
            effect_ends.append( 0 )
            
    tce_pdf = tce_pdf.with_columns(
        pl.Series(name="cause_start", values=cause_starts),
        pl.Series(name="cause_end", values=cause_ends),
        pl.Series(name="effect_start", values=effect_starts),
        pl.Series(name="effect_end", values=effect_ends)
    )            
    
    return tce_pdf



def ce_token_start_end_pos(tce_pdf:pl.DataFrame):
    
    '''
    tce_pdf:
    
        Text	Cause	Effect	cause_start	cause_end	effect_start	effect_end
        str	str	str	i64	i64	i64	i64
        "It found that total U.S. healt…	"Part of the reason is that Med…	"It found that total U.S. healt…	163	321	0	162
        "Transat loss more than doubles…	"it works to complete Air Canad…	"Transat loss more than doubles"	34	70	0	30     
    
    
    returns:
    
        Text	Cause	Effect	cause_start	cause_end	effect_start	effect_end	c_tok_start	c_tok_end	e_tok_start	e_tok_end
        str	str	str	i64	i64	i64	i64	i64	i64	i64	i64
        "It found that total U.S. healt…	"Part of the reason is that Med…	"It found that total U.S. healt…	163	321	0	162	32	60	0	32
        "Transat loss more than doubles…	"it works to complete Air Canad…	"Transat loss more than doubles"	34	70	0	30	6	13	0	5    
    
    '''
    
    cause_tok_starts = []
    cause_tok_ends = []
    effect_tok_starts = []
    effect_tok_ends = []
    for i, (t, c, e, cs, ce, es, ee) in tqdm(
            enumerate(tce_pdf.iter_rows()),
            desc='Get Cause Effect Start/End positions.'
        ):

        if isinstance(c, str):

            cs_tokens = nlp(t[:cs])
            cause = nlp(c)

            cs_start = len(cs_tokens)
            cause_tok_starts.append( cs_start )
            cause_tok_ends.append( cs_start + len(cause) )

        else:

            cause_tok_starts.append( 0 )
            cause_tok_ends.append( 0 )        

        if isinstance(e, str):

            es_tokens = nlp(t[:es])
            effect = nlp(e)

            es_start = len(es_tokens)
            effect_tok_starts.append( es_start )
            effect_tok_ends.append( es_start + len(effect) )

        else:

            effect_tok_starts.append( 0 )
            effect_tok_ends.append( 0 )
            
    tce_pdf = tce_pdf.with_columns(
        pl.Series(name="c_tok_start", values=cause_tok_starts),
        pl.Series(name="c_tok_end", values=cause_tok_ends),
        pl.Series(name="e_tok_start", values=effect_tok_starts),
        pl.Series(name="e_tok_end", values=effect_tok_ends)
    )

    return tce_pdf



def ce_generate_samples(tce_pdf:pl.DataFrame):
    
    '''
    
    tce_pdf:
    
        Text	Cause	Effect	cause_start	cause_end	effect_start	effect_end	c_tok_start	c_tok_end	e_tok_start	e_tok_end
        str	str	str	i64	i64	i64	i64	i64	i64	i64	i64
        "It found that total U.S. healt…	"Part of the reason is that Med…	"It found that total U.S. healt…	163	321	0	162	32	60	0	32
        "Transat loss more than doubles…	"it works to complete Air Canad…	"Transat loss more than doubles"	34	70	0	30	6	13	0	5
    
    '''

    docs = []
    cause_bio_docs = [] # 0:B, 1:I, 2:O
    effect_bio_docs = [] # 0:B, 1:I, 2:O
    cause_effect_bio_docs = [] # 0:B-C, 1:I-C, 2:0, 3:B-E, 4:I-E
    for t, cts, cte, ets, ete in tqdm(tce_pdf[[
            'Text', 'c_tok_start', 'c_tok_end', 'e_tok_start', 'e_tok_end'
        ]].iter_rows(), desc='token labels'):

        docs.append(nlp(t))
        cause_bio_docs.append([])
        effect_bio_docs.append([])
        cause_effect_bio_docs.append([])

        for i in range(len(docs[-1])):

            if i == cts:
                cause_bio_docs[-1].append( 0 )
                cause_effect_bio_docs[-1].append( 0 )
            elif i > cts and i < cte:
                cause_bio_docs[-1].append( 1 )
                cause_effect_bio_docs[-1].append( 1 )
            else:
                cause_bio_docs[-1].append( 2 )
                if i == ets:
                    cause_effect_bio_docs[-1].append( 3 )
                elif i > ets and i < ete:
                    cause_effect_bio_docs[-1].append( 4 )      
                else:
                    cause_effect_bio_docs[-1].append( 2 )

            if i == ets:
                effect_bio_docs[-1].append( 0 )
            elif i > ets and i < ete:
                effect_bio_docs[-1].append( 1 )
            else:
                effect_bio_docs[-1].append( 2 )
                
    return docs, cause_bio_docs, effect_bio_docs, cause_effect_bio_docs