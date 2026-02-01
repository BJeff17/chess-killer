"""
Coder les regles du jeux et le jeu en lui meme
    -mouvement des pieces sans les restrictions
    -condition de fin d'une partie

"""
import copy


class ChessBoard:

    def __init__(self,white, black):

        """
            L'échiquier comportera les pieces qui seront des instanes enfants de Piece,
            le vide sera aussi une piece: piece nulle.
            Il ne faut pas oublier de faire une representation numérique pour mon(mes) reseau(x) de neurone...
        """
        
        self.white, self.black = white, black
        self.board = [[],[],[],[],[],[],[],[]]
        self.lastmove = (-1,-1)
        self.WhitePiece = self.getOpponentPiece(1)
        self.BlackPiece = self.getOpponentPiece(-1)

    def __getitem__(self, key):
        return self.board[key]

    def getOpponentPiece(self,color):
        opponents:list[Piece] = []
        for i in self.board:
            for j in i:
                if i.color == - color:
                    opponents.add(i)
        return opponents
    
    def isCaseControl(self,case,color):
        x,y = case
        opponents = self.WhitePiece if color == -1  else self.BlackPiece
        for i in opponents:
            if i.IsvalidDep(self,case ):return True            
        return False
    def setMove(self,move):
        x,y,x_,y_ = move
        self.tempPiece = self.board[x_][y_]
        self.board[x][y], self.board[x_][y_] = BlankPiece(),self.board[x][y]
        self.board[x_][y_].setMove(self,(x_,y_))
        self._lastmove = self.lastmove.copy()
        self.lastmove = move
    def discard(self, move):
        x,y,x_,y_ = move
        self.board[x][y], self.board[x_][y_] = self.board[x_][y_],self.tempPiece
        self.board[x_][y_].setMove(self,(x_,y_))
        self.lastmove = self._lasmove


    def allow(self,move):
        x,y,x_,y_ = move
        if self[x][y].IsvalidDep(self, (x_,y_)) :
            self.setMove(move)
            if self.echec(self[x][y].color):
                self.discard(move)
                return False
            self.discard(move)
            return True
        return False
             
    def echec(self,color):
        FriendPiece = self.WhitePiece if color == 1 else self.BlackPiece
        for i in FriendPiece:
            if isinstance(i, King):
                if self.isCaseControl(i.currentPos,color):
                    return True
                return False
                
    def gridState(self):
        pass

class Piece:
    def __init__(self, id:int, color:int):#color 1->White -1-> Black 0->blank
        self.valid_dep = []#[(dx,dy),...]
        self.id = id
        self.color = color
        self.isFirstMove = True
        self.currentPos = (-1,-1)
        self.isBlank = False

    def IsvalidDep(self, board, move):#move:(x,y)
        x_,y_,x,y = self.currentPos, move
        if x<0 or x>7 or y>7 or y<0:return False
        isGoodDir = False
        for d in self.valid_dep:
            for i in range(0,8):
                t,z = i*d[0]+x_,i*d[1]+y_
                if x == t and y == z:
                    isGoodDir = True
                if isinstance(board[t][z],BlankPiece)  and t in range(x,x_, -1 if x-x_>0 else 1) and z in range(y,y_, -1 if y-y_>0 else 1) and (x != t or y != z):
                    return False
            if isGoodDir :
                return True
        return False
    def IsvalidDep1(self,board, move):
        if x<0 or x>7 or y>7 or y<0:return False
        x_,y_,x,y = move["x_"], move["y_"], move["x"], move["y"]
        for d in self.valid_dep:
            t,z = d[0]+x_,d[1]+y_
            if x == t and y == z:
                return True
        return False
    def setMove(self,board,move):
        if self.IsvalidDep(board,move) and board.allow(self.currentPos, move):
            self.currentPos = move
            self.isFirstMove = False

class BlankPiece(Piece):
    def __init__(self):
        super().__init__(id,0,0)
        self.isBlank = True             

"""
    C'est l'heure de faire les divers type de piece
    King # ne pas oublié d'implementer le roque.
    Queen
    Knight
    Bishop
    Rook
    Pown # ne pas oublier d'implementer en passant
"""

class Rook(Piece):
    def __init__(self, id, color):
        super().__init__(id, color)
        self.valid_dep = [(-1,0),(0,-1), (1,0),(0,1)]#Se deplace suivant les lignes et colonnes

class Bishop(Piece):
    def __init__(self, id, color):
        super().__init__(id, color)
        self.valid_dep = [(-1,-1),(-1,1), (1,-1),(1,1)]#Se deplace suivant les diagonales

class Queen(Piece):
    def __init__(self, id, color):
        super().__init__(id, color)
        self.valid_dep = [(-1,0),(0,-1), (1,0),(0,1),(-1,-1),(-1,1), (1,-1),(1,1)]#Se deplace suivant les lignes et colonnes et les diagonales
#C'etait les plus facile on passe au piece les plus étranges(il ne faut ps que j'oublie d'implementer le roque)
"""
En fait ces pieces n'avance que d'une case ou si vous preferez sont de courte porté
"""
class Knight(Piece):
    def __init__(self, id, color):
        super().__init__(id, color)
        self.valid_dep = [(2,1),(2,-1), (1,2),(1,-2),(-2,1),(-2,-1),(-1,2),(-1,-2)]#L'octopus il avance en L (ne dite pas à mes pote que j'ai dis qu'il avance en L...)
    def IsvalidDep(self, board, move):
        return super().IsvalidDep1(board, move)
    
class King(Piece):
    def __init__(self, id, color):
            super().__init__(id, color)
            self.valid_dep = [(-1,0),(0,-1), (1,0),(0,1),(-1,-1),(-1,1), (1,-1),(1,1),]#Il avance d'une case autour de lui
    
    def isRoque(self, move):
        if not self.isFirstMove:return 0
        if (self.color>0 and move[0] == 7 and move[1] == 6) or (self.color<0 and move[0] == 0 and move[1] == 6): return 1#petit roque
        if (self.color>0 and move[0] == 7 and move[1] == 2) or (self.color<0 and move[0] == 0 and move[1] == 2): return -1#grand roque
        return 0
    
    def IsvalidDep(self, board:ChessBoard, move):
        if super().IsvalidDep1(board, move) and not board.isCaseControl(move[0],move[1],-self.color):return True
        isRoque = self.isRoque(move)
        if isRoque == 1:#petit roque
            rook:Rook|BlankPiece = board[7][7] if self.color>0 else board[0][7]#il dfaut bien verifier les tours des bonne couleur
            if rook.isBlank or not rook.isFirstMove:return False
            #Je voulais retourner True mais je me suis rendu compte que si des cas sont controlé on ne peut pas roquer(decidement, roquer c'est un mot dure à ecrire et dure à implmenter...)
            IsControled = False
            for i in range(2):
                IsControled = IsControled or board.isCaseControl(move[0],move[1]+i,-self.color)#someone should add or= in python...
            return not IsControled
        if isRoque == -1:#grand roque
            rook:Rook|BlankPiece = board[7][0] if self.color>0 else board[0][0]#il dfaut bien verifier les tours des bonne couleur
            if rook.isBlank or not rook.isFirstMove:return False
            #Je voulais retourner True mais je me suis rendu compte que si des cas sont controlé on ne peut pas roquer(decidement, roquer c'est un mot dure à ecrire...)
            IsControled = False
            for i in range(2):
                IsControled = IsControled or board.isCaseControl(move[0],move[1]-i,-self.color)#someone should add or= in python maybe i will contribute and hope someone will accept my commit(nah!!)...Il ne faut pas oublier de fournir la methode isCasControl
            return not IsControled

class Pown(Piece):
    
    def __init__(self, id, color):
        super().__init__(id, color)
        #Eux ne prennent pas de la meme manier qu'il se deplace, il faut donc verifier s'il s'agit d'une prise ou d'un déplacement d'ailleur au premier coups ils avancent d'une ou deux case au choix et arrivé en dernier rangé ils deviennent ce qu'ils veulent
        self.personality = None #Il ne faut pas faire la gaffe d'initialiser sinon, on aura une boucle infinie cet attribut servira en cas de promotion.

    def IsvalidDep(self, board:ChessBoard, move):
        if self.personality is None:
            x,y,x_,y_ = self.currentPos, move
            piece = board[x_][y_]
            if x-piece.color==x_ and (y == y_+1 or y==y_-1) and not piece.isBlank:
                return True
                    
            if piece.isBlank and not board[x_][y_-self.color].isBlank:
                piece = board[x_][y_-self.color]
                if isinstance(piece,Pown) and piece.color == - self.color:
                    if x_ == x and (y_ == y+1 or y_==y-1) and board.lastmove == (x_-2,y_,x_,y_-self.color):
                        return True#en passant
                return False
            if self.isFirstMove and x == x_ + self.color*2 and y==y_: return True
            if x == x_ + self.color and y==y_ and not board[x_][y_].isBlank:return True
        else: 
            return self.personality.IsvalidDep(board, move)#je n'aime pas trop ce que je fais ici parce que cà implique que je dois initialiser une instance de la classe vers laquelle on promeut le pion mais bon, on ne promeut qu'une fois au plus il n'y a que 8 promotions possibles
    
    def promotion(self, board:ChessBoard, move, newPersonality:Piece):
        if self.IsvalidDep(board, move) and move[0] == 7*int((1 -self.color)/2):
            self.personality = newPersonality()
            self.personality.currentPos = move
            self.personality.color = self.color
            
            

