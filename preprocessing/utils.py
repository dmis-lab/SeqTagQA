from six import string_types

def unnesting_list(inp): 
    unnestedList=list()
    
    if type(inp)==list:
        for subEle in inp:
            unnestedList+=unnesting_list(subEle)
        return unnestedList
    elif isinstance(inp, string_types): # python3.6
        return [inp]
    else:
        try:
            int(inp) # numeric
            return [inp]
        except:
            print("FATAL : check input type! not a string, not a numeric value")
            raise

            
def smartLower(answer):
    """
    Lowercased comparison, but rule-based exceptions are applied 
       to handle some entities, such as BRCA1,... , by comparing them as cased comparison
    """
    if len(answer)==1:
        return answer
    ansList = answer.split()
    for subans in ansList:
        if sum(1 for c in subans[1:] if c.isupper()) > 0: # should be c.isupper() -> 20201029 update
            return answer
    return answer.lower()


def normalize_unicode(text): 
    # The name was func preprocess from BERN
    #
    # From http://jkorpela.fi/chars/spaces.html  
    #   AND  https://www.fileformat.info/info/unicode/version/1.1/index.htm
    #
    
    # Line chars
    text = text.replace('\n', ' ') # added 20201030
    text = text.replace('\r', ' ')
    text = text.replace('\u2028', ' ')
    text = text.replace('\u2029', ' ')
    
    # QUADs
    text = text.replace('\u2000', ' ')
    text = text.replace('\u2001', ' ')

    # EN SPACE
    # https://www.fileformat.info/info/unicode/char/2002/index.htm
    text = text.replace('\u2002', ' ')
    
    # FOUR-PER-EM SPACE
    # https://www.fileformat.info/info/unicode/char/2005/index.htm
    text = text.replace('\u2003', ' ')
    text = text.replace('\u2004', ' ')
    text = text.replace('\u2005', ' ')
    text = text.replace('\u2006', ' ')
    
    # FIGURE SPACE
    text = text.replace('\u2007', ' ')

    # THIN SPACE
    # https://www.fileformat.info/info/unicode/char/2009/index.htm
    text = text.replace('\u2008', ' ')
    text = text.replace('\u2009', ' ')
    
    # HAIR / ZERO SPACE
    # https://www.fileformat.info/info/unicode/char/200a/index.htm
    text = text.replace('\u200A', ' ')
    text = text.replace('\u200B', ' ')

    # NO-BREAK SPACE
    # https://www.fileformat.info/info/unicode/char/202f/index.htm
    text = text.replace('\u202F', ' ')
   
    # https://www.fileformat.info/info/unicode/char/00a0/index.htm
    text = text.replace('\u00A0', ' ')

    # https://www.fileformat.info/info/unicode/char/f8ff/index.htm
    text = text.replace('\uF8FF', ' ')
    
    text = text.replace('\uFEFF', ' ')
    text = text.replace('\uF044', ' ')
    text = text.replace('\uF02D', ' ')
    text = text.replace('\uF0BB', ' ')

    text = text.replace('\uF048', 'Η')
    text = text.replace('\uF0B0', '°')
    
    # MATH SPACE 
    text = text.replace('\u205F', ' ')
    
    # ETCs SPACE http://jkorpela.fi/chars/spaces.html
    text = text.replace('\u1680', ' ')
    text = text.replace('\u180E', ' ')
    text = text.replace('\u1680', ' ')
    text = text.replace('\u3000', ' ')

    # MIDLINE HORIZONTAL ELLIPSIS: ⋯
    # https://www.fileformat.info/info/unicode/char/22ef/index.htm
    # text = text.replace('\u22EF', '...')
    
    # PUNCTUATION, DASH
    # https://www.fileformat.info/info/unicode/category/Pd/list.htm
    
    text = text.replace('\u2012', '-')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u2015', '-')
    text = text.replace('\u2E3A', '-')
    text = text.replace('\u2E3B', '-')
    text = text.replace('\uFE58', '-')

    # for list-seq task : 20201030
    text = text.replace('\t', ' ')
    #text = text.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
    
    return text

def find_all_str(answer, context, casing = "Force"):
    """
    casing : Cased | Force | Intelli
        Cased means that the true case and accent markers are preserved when strings are compared. 
        Force means that the words are lower cased and compared.
        Inteli is based on lowercased comparison but rule-based exceptions are applied 
           to handle some entities, such as BRCA1,... , by comparing them as cased comparison.
    """
    if casing == "Cased":
        answerCase = answer
        contextCase = context
    elif casing == "Force":
        answerCase = answer.lower()
        contextCase = context.lower()
    elif casing == "Intelli":
        answerCase = answer.lower()
        contextCase = context.lower()
        
        # exceptional case : One char
        if len(answer)==1:
            answerCase = answer
            contextCase = context
        # exceptional case : Named Entity such as BRCA
        ansList = answer.split()
        for subans in ansList:
            if sum(1 for c in subans[1:] if c.isupper()) > 0:
                answerCase = answer
                contextCase = context
                break
                
        # hard intelli : matching entities by checking seperation. 
        # ex) Looking for CMT => EgCMT1 (x) | XYZ, CMT, ABC, ... (o)  
        # special char : spacing, comma, period, (, ),  
        if True: # for entity search
            specialChar = ".,:;()/!?'$'" + '"' # updated 20201030
            for schar in specialChar:
                if schar in answer: # exclude that char from spcial char list
                    specialChar = specialChar.replace(schar, "")
                    
            newContextList = []
            for char in contextCase:
                if char in specialChar:
                    newContextList.append(" ") # ex) CMT, ABC -> CMT  ABC
                else:
                    newContextList.append(char)
            contextCase = " " + "".join(newContextList) + " "
            answerCase = " "+answerCase+" "
            
    else:
        raise NotImplementedError
        
    idx = contextCase.find(answerCase)
    while idx != -1:
        yield idx
        idx = contextCase.find(answerCase, idx+1)