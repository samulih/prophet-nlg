import logging
from typing import Dict, List
from uralicNLP.ud_tools import UD_node

logger = logging.getLogger(__name__)

UPOS_TO_POS_MAP = {
    'NOUN': 'N',
    'PROPN': 'N',
    'ADJ': 'A',
    'VERB': 'V',
    'AUX': 'V',
    'CCONJ': 'CC',
    'SCONJ': 'CS',
    'ADP': 'Adp',
    'ADV': 'Adv',
    'PRON': 'Pron',
    'PUNCT': 'Punct',
    'SYM': 'Punct',
    'INTJ': 'Interj',
    'NUM': 'Num',
    'DET': 'Pron',
    'X': 'Forgn'
}

tags = [
    'N', 'A', 'Num', 'Pron', 'V', 'Adv', 'Adp', 'Pcle', 'Interj',
    'CC', 'CS',
    'Prop',
    # Derivations
    'Der/minen', 'Der/maisilla', 'Der/ja', 'Der/tar', 'Der/ttaa', 'Der/tattaa',
    'Der/tatuttaa', '+Der/u', '+Der/inen', '+Der/llinen',
    # Adverbs and adpositions
    'Prl', 'Dis',
    'Po', 'Prs',
    # Adjectives
    'Comp', 'Superl',
    'Card', 'Ord', 'Rom',
    # Verbs
    'Neg',
    'Act', 'Pss',
    'InfA', 'InfE', 'InfMa', 'Ind', 'Imprt', 'Cond', 'Pot',
    'Prs', 'Prt', 'PrfPrc', 'PrsPrc', 'ConNeg', 'NegPrc', 'AgPrc',
    # Pronouns
    'Pers', 'Dem', 'Interr', 'Rel', 'Qua', 'Reflex', 'Recipr',
    # Singular/plural
    'Sg', 'Pl',
    'Sg1', 'Sg2',' Sg3', 'Pl1', 'Pl2', 'Pl3',
    # Nominal declination
    'Nom', 'Par', 'Acc', 'Gen', 'Ine', 'Ela', 'Ill', 'Ade', 'Abl', 'All', 'Ess', 'Tra', 'Abe', 'Com',
    # Verb possessive suffixes
    'PxSg1', 'PxSg2', 'PxSg3', 'PxPl1', 'PxPl2', 'PxPl3', 'Px3', 'Pe4',
    # Question and focus particles
    'Qst', 'Foc/pa', 'Foc/s', 'Foc/ka', 'Foc/han', 'Foc/kin', 'Foc/kaan'
]

tag_indexes = {tag: idx for idx, tag in enumerate(tags)}


# NOTE: taken originally from Omorfi `get_ftb_feats`
# modified to not exit on error, plus changed the logic to
# have deterministic output regards to feat order
def get_morphology_parts(node: UD_node) -> List[str]:
    lemma, pos = node.lemma, node.pos
    feats = dict(f.split('=', 1) for f in node.get_feats() if '=' in f)
    misc = dict(f.split('=', 1) for f in node.misc.split('|') if '=' in f)

    rvs = list()
    rvs += [UPOS_TO_POS_MAP.get(pos, 'Unkwn')]
    if pos == 'PROPN':
        rvs += ['Prop']
    elif pos == 'ADV' and lemma.endswith("sti"):
        # This is FTB oddity
        rvs += ['Pos', 'Man']
    for key, value in feats.items():
        if key == 'Number':
            if value == 'Sing':
                rvs += ['Sg']
            elif value == 'Plur':
                rvs += ['Pl']
        elif key == 'Tense':
            if value == 'Pres':
                rvs += ['Prs']
            elif value == 'Past':
                rvs += ['Prt']
        elif key == 'Mood':
            if value == 'Ind':
                continue
            elif value == 'Cnd':
                rvs += ['Cond']
            elif value == 'Impv':
                rvs += ['Imp']
            else:
                rvs += [value]
        elif key == 'Voice':
            rvs += [value]
        elif key == 'Person':
            if value == '0':
                rvs += ['__3']
            elif value == '1':
                rvs += ['__1']
            elif value == '2':
                rvs += ['__2']
            elif value == '3':
                rvs += ['__3']
            elif value == '4':
                rvs += ['Pe4']
            else:
                logger.warning(key, value, "for ftb")
        elif key == 'Number[psor]':
            if value == 'Sing':
                rvs += ['PxSg_']
            elif value == 'Plur':
                rvs += ['PxPl_']
            else:
                logger.warning(key, value, "for ftb")
        elif key == 'Person[psor]':
            if value == '1':
                rvs += ['Px__1']
            elif value == '2':
                rvs += ['Px__2']
            elif value == '3':
                rvs += ['PxSp3']
            else:
                logger.warning(key, value, "for ftb")
        elif key == 'Polarity':
            if value == 'Neg':
                rvs += ['Neg']
            else:
                logger.warning(key, value, "for ftb")
        elif key == 'Connegative':
            if value == 'Yes':
                rvs += ['Act', 'ConNeg']
            else:
                logger.warning(key, value, "for ftb")
        elif key == 'InfForm':
            if value == '1':
                rvs += ['Inf1', 'Lat']
            elif value == '2':
                rvs += ['Inf2']
            elif value == '3':
                rvs += ['Inf3']
            elif value == 'MINEN':
                rvs += ['Inf4']
            elif value == 'MAISILLA':
                rvs += ['Inf5']
        elif key == 'PartForm':
            # FTB participle is POS
            pass
        elif key == 'Case':
            rvs += [value]
        elif key == 'Degree':
            rvs += [value]
        elif key == 'SUBCAT':
            if value == 'NEG':
                rvs += ['Neg']
            elif value == 'QUOTATION':
                rvs += ['Quote']
            elif value == 'QUANTIFIER':
                rvs += ['Qnt']
            elif value == 'DIGIT':
                rvs += ['Digit']
            elif value in ['COMMA', 'BRACKET',
                            'ARROW', 'DECIMAL', 'PREFIX', 'SUFFIX']:
                # not annotated in FTN feats:
                # * punctuation classes
                continue
            elif value == 'ROMAN':
                # not annotated in FTN feats:
                # * decimal, roman NumType
                continue
            else:
                logger.warning(key, value, "SUBCAT")
        elif key == 'NumType':
            if value == 'Ord':
                rvs += [value]
            else:
                pass
        elif key == 'PronType':
            if value == 'Prs':
                rvs += ['Pers']
            elif value == 'Ind':
                rvs += ['Qnt']
            elif value == 'Int':
                rvs += ['Interr']
            else:
                rvs += [value]
        elif key == 'AdpType':
            if value == 'Post':
                rvs += ['Po']
            elif value == 'Prep':
                rvs += ['Pr']
            else:
                logger.warning(key, value, 'ADPTYPE', 'FTB3')
        elif key == 'Clitic':
            if value == 'Ka':
                rvs += ['Foc_kA']
            else:
                rvs += ['Foc_' + value]
        elif key == 'Abbr':
            rvs += ['Abbr']
        elif key == 'Derivation':
            if value in ['NUT', 'VA']:
                rvs += ['Act']
            elif value in ['TU', 'TAVA']:
                rvs += ['Pss']
            else:
                continue
        elif key == 'Reflex':
            rvs += ['Refl']
        elif key in ['UPOS', 'ALLO', 'WEIGHT', 'CASECHANGE', 'NEWPARA',
                        'GUESS', 'PROPER', 'SEM', 'CONJ', 'BOUNDARY',
                        'PCP', 'DRV', 'LEX', 'BLACKLIST', 'Style',
                        'POSITION', "Foreign", 'VerbForm',
                        'Typo']:
            continue
        else:
            logger.warning(key, value, 'FTB3')
    for key, value in misc.items():
        if key == 'NumType':
            rvs += [value]
        elif key == 'Person' and value == '4':
            rvs += ['Pe4']
        elif key == 'PunctType':
            if value == "Quotation":
                rvs += ["Quote"]
            elif value == "Dash":
                if lemma == '—':
                    rvs += ['EmDash']
                elif lemma == '–':
                    rvs += ['EnDash']
                else:
                    rvs += ['Dash']
            elif value in ["Comma", "Bracket", "Arrow"]:
                pass
            else:
                logger.warning(key, value, 'FTB3')
        elif key == 'PropnType':
            rvs += ["Prop"]
        elif key in ['AffixType', "GoesWith", "Position"]:
            # XXX
            pass
        elif key == 'SemType':
            pass
        elif key == "Derivation":
            if value in ["Tava", "Tu"]:
                rvs += ["Pass"]
            elif value in ["Va", "Nut"]:
                rvs += ["Act"]
            else:
                pass
        elif key == "Mood":
            if value == 'Opt':
                rvs += ["Opt"]
        elif key in ["Lexicalised", "Blacklisted"]:
            continue
        # ignore unknown misc
    # post hacks
    if lemma == 'ei' and 'Foc_kA' in rvs:
        revs = []
        for r in rvs:
            if r != 'V':
                revs += [r]
            else:
                revs += ['CC']
        rvs = revs
    if 'Punct' in rvs and 'Sg' in rvs and 'Nom' in rvs:
        revs = []
        for r in rvs:
            if r not in ['Sg', 'Nom']:
                revs += [r]
        rvs = revs
    if '__1' in rvs or '__2' in rvs or '__3' in rvs:
        revs = []
        for r in rvs:
            if r not in ['__1', '__2', '__3', 'Sg', 'Pl']:
                revs += [r]
        if 'Sg' in rvs and '__1' in rvs:
            revs += ['Sg1']
        elif 'Sg' in rvs and '__2' in rvs:
            revs += ['Sg2']
        elif 'Sg' in rvs and '__3' in rvs:
            revs += ['Sg3']
        elif 'Pl' in rvs and '__1' in rvs:
            revs += ['Pl1']
        elif 'Pl' in rvs and '__2' in rvs:
            revs += ['Pl2']
        elif 'Pl' in rvs and '__3' in rvs:
            revs += ['Pl3']
        else:
            logger.warning("__X without Sg or Pl")
        rvs = revs
    if 'Px__1' in rvs or 'Px__2' in rvs or 'Px__3' in rvs:
        revs = []
        for r in rvs:
            if r not in ['Px__1', 'Px__2', 'Px__3', 'PxSg_', 'PxPl_']:
                revs += [r]
        if 'PxSg_' in rvs and 'Px__1' in rvs:
            revs += ['PxSg1']
        elif 'PxSg_' in rvs and 'Px__2' in rvs:
            revs += ['PxSg2']
        elif 'PxSg_' in rvs and 'Px__3' in rvs:
            revs += ['PxSg3']
        elif 'PxPl_' in rvs and 'Px__1' in rvs:
            revs += ['PxPl1']
        elif 'PxPl_' in rvs and 'Px__2' in rvs:
            revs += ['PxPl2']
        elif 'PxPl_' in rvs and 'Px__3' in rvs:
            revs += ['PxPl3']
        elif 'Px__3' in rvs:
            revs += ['Px3']
        else:
            logger.warning("__X without Sg or Pl")
        rvs = revs
    if 'Neg' in rvs and 'Act' in rvs:
        revs = []
        for r in rvs:
            if r != 'Act':
                revs += [r]
        rvs = revs
    if 'Abbr' in rvs:
        revs = []
        for r in rvs:
            if r not in ['N', 'Prop']:
                revs += [r]
        rvs = revs
    if 'Inf1' in rvs:
        revs = []
        for r in rvs:
            if r not in ['Act', 'Pl', 'Sg']:
                revs += [r]
        rvs = revs
    if 'Pers' in rvs:
        revs = []
        for r in rvs:
            if r not in ['Pl1', 'Sg1', 'Pl2', 'Sg2', 'Pl3', 'Sg3']:
                revs += [r]
            elif r in ['Pl1', 'Pl2', 'Pl3']:
                revs += ['Pl']
            elif r in ['Sg1', 'Sg2', 'Sg3']:
                revs += ['Sg']
            else:
                logger.warning(revs, r)
        rvs = revs
    if 'Card' in rvs and 'Digit' in rvs:
        revs = []
        for r in rvs:
            if r != 'Card':
                revs += [r]
        rvs = revs

    order = tag_indexes
    return sorted(rvs, key=lambda x: order.get(x, 15))


def ud_node_morphology(node: UD_node) -> str:
    feats = '+'.join(get_morphology_parts(node))
    return f'{node.lemma}+{feats}'
