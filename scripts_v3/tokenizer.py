from . import lm
from . import mdp
import unicodedata

# each method name is correlated to sentencepiece
class Tokenizer:
    def __init__(self, mlm):
        self.mlm = mlm
        self.id_to_piece = self.mlm.id_to_piece
        self.piece_to_id = self.mlm.piece_to_id
        self.wordPieceMode = hasattr(self.mlm, 'wordPiecePrefix') and self.mlm.wordPiecePrefix is not None
    
    def encode_as_pieces(self, line):
        return self.mlm.convertIds2Words(self.encode_as_ids(line))
    
    def encode_as_ids(self, line):
        idTable = self.mlm.makeIdTable(line)
        logProbTable = self.mlm.makeLogProbTable(line, idTable=idTable)
        ids = mdp.viterbiIdSegmentation(idTable, logProbTable)
        return ids

    def sample_encode_as_pieces(self, line, n=-1, alpha=0.2):
        return self.mlm.convertIds2Words(self.sample_encode_as_ids(line, n, alpha))
    
    def sample_encode_as_ids(self, line, n=-1, alpha=0.2):
        idTable = self.mlm.makeIdTable(line)
        logProbTable = self.mlm.makeLogProbTable(line, idTable=idTable)
        if 0<n:
            ids = mdp.mSampleFromNBestIdSegmentation(idTable, logProbTable, 1, n, mode='astar', lam=alpha)[0]
        else:
            # FFBS
            logProbTable *= alpha
            ids = mdp.samplingIdSegmentation(idTable, logProbTable)
        return ids

    def nbest_encode_as_pieces(self, line, n):
        return [self.mlm.convertIds2Words(ids) for ids in self.nbest_encode_as_ids(line, n)]
    
    def nbest_encode_as_ids(self, line, n):
        idTable = self.mlm.makeIdTable(line)
        logProbTable = self.mlm.makeLogProbTable(line, idTable=idTable)
        return mdp.nbestIdSegmentation(idTable, logProbTable, n)

    def get_piece_size(self):
        return len(self.mlm.vocab) 

