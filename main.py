import math
import pprint
import random
import torch
import torch.nn as nn
from torch import  optim 
import copy
"""
1  -> pown
KNIGHT-> knight
BISHOP  -> bishop
ROOK  -> rook
QUEEN  -> queen
KING -> king

"""




WHITE = 1
BLACK = -1
NULL = 2

KING = 20
POWN = 1
KNIGHT = 2.7
BISHOP = 3
ROOK = 5
QUEEN = 9
PLACEHOLDER = 0.5

abs = lambda x:-x if x<0 else x

directionGrid = {
    POWN  : [(-1,1),(1,1)],
    KNIGHT: [(1,2),(-1,2),(1,-2),(-1,-2),(2,1),(-2,1),(2,-1),(-2,-1)],
    BISHOP:[(1,1),(-1,1),(1,-1),(-1,-1)],
    ROOK:[(1,0),(0,1),(0,-1),(-1,0)],
    PLACEHOLDER:[],
    QUEEN:[(1,0),(0,1),(0,-1),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)],
    KING: [(1,1),(-1,-1),(1,-1),(-1,1),(1,0),(0,1),(-1,0),(0,-1)]

}


validMoveGrid = {
    POWN  : [(0,1),(0,2),(-1,1),(1,1)],
    KNIGHT: [(1,2),(-1,2),(1,-2),(-1,-2),(2,1),(-2,1),(2,-1),(-2,-1)],
    BISHOP  : [(i,i) for i in range(8)] + [(-i,i) for i in range(8)] + [(i,-i) for i in range(8)] + [(-i,-i) for i in range(8)],
    ROOK  : [(i,0) for i in range(8)] + [(0,i) for i in range(8) ] + [(0,-i) for i in range(8) ] + [(-i,0) for i in range(8) ],
    QUEEN  : [(i,i) for i in range(8)] + [(-i,i) for i in range(8)] + [(-i,i) for i in range(8)] + [(i,0) for i in range(8)] + [(0,i) for i in range(8) ] + [(0,-i) for i in range(8) ] + [(-i,0) for i in range(8) ],
    KING: [(1,1), (-1,1), (1,-1), (-1,1), (1,0), (0,1), (-1,0), (0,-1)]
}



INITCHESSGAME = [
    [BLACK*ROOK,BLACK*KNIGHT,BLACK*BISHOP,BLACK*QUEEN,BLACK*KING,BLACK*BISHOP,BLACK*KNIGHT,BLACK*ROOK],
    [BLACK*POWN,BLACK*POWN,BLACK*POWN,BLACK*POWN,BLACK*POWN,BLACK*POWN,BLACK*POWN,BLACK*POWN],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [WHITE*POWN,WHITE*POWN,WHITE*POWN,WHITE*POWN,WHITE*POWN,WHITE*POWN,WHITE*POWN,WHITE*POWN],
    [WHITE*ROOK,WHITE*KNIGHT,WHITE*BISHOP,WHITE*QUEEN,WHITE*KING,WHITE*BISHOP,WHITE*KNIGHT,WHITE*ROOK],
]


chess_board_pos = [
    *[BLACK*ROOK,0,BLACK*BISHOP,0,0,BLACK*ROOK,BLACK*KING,0],
    *[BLACK*POWN,0,BLACK*QUEEN,0,0,BLACK*POWN,BLACK*POWN,BLACK*POWN],
    *[0,BLACK*POWN,0,0,0,0,0,0],
    *[0,0,0,0,0,0,0,0],
    *[0,0,0,0,0,0,0,0],
    *[WHITE*QUEEN,WHITE*POWN,0,0,0,WHITE*KNIGHT,0,0],
    *[WHITE*POWN,0,0,0,WHITE*ROOK,WHITE*POWN,WHITE*POWN,WHITE*POWN],
    *[0,0,0,0,WHITE*ROOK,0,WHITE*KING,0],
]


initChessbar = lambda x,y:[[0]*y]*x
flatened_chess_board = torch.tensor(chess_board_pos)

class MyChessMaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(64+15,32) # AX +B
        self.L2 = nn.Linear(32,64)
        self.LBISHOP = nn.Linear(64,64)
        self.out = nn.Linear(64,1)
    def forward(self,x):
        out = self.L1(x)
        out = self.L2(out)
        out = self.LBISHOP(out)
        return nn.functional.tanh(self.out(out))
    def eval_(self,board, lastmoves): 
        return self.forward(torch.tensor([j for i in board for j in i]+[j for i in lastmoves for j in i], dtype=torch.float32))

class MyChessMaster2(nn.Module):
    def __init__(self):
        super().__init__()
        # self.L1 = nn.Linear(64,32)
        # self.L2 = nn.Linear(32,64)
        self.out = nn.Linear(64,1)
    def forward(self,x):
        out = x
        # out = self.L1(x)
        # out = self.L2(out)
        return nn.functional.tanh(self.out(out))
    def eval_(self,board): return self.forward(torch.tensor([j for i in board for j in i]))

engine1 = MyChessMaster()
engine2  = MyChessMaster()


def printGame(board):
    
    for i in board:
        print("| ",end="")
        for j in i:
            if j==0:print("   | ", end="")
            flag = "b" if j < 0 else "w"
            if abs(j) == POWN: print(f"P{flag}", end=" | ")
            elif abs(j) == KNIGHT: print(f"C{flag}",end=" | ")
            elif abs(j) == BISHOP: print(f"F{flag}",end=" | ")
            elif abs(j) == ROOK: print(f"T{flag}",end=" | ")
            elif abs(j) == QUEEN: print(f"D{flag}",end=" | ")
            elif abs(j) == KING : print(f"R{flag}", end=" | ")
        print(f"\n")



def canTake(board, attacker_x, attacker_y, piece_x, piece_y, lastmove):
    attacker = board[attacker_x][attacker_y]

    if abs(attacker) == PLACEHOLDER: return False
    if attacker * board[piece_x][piece_y] > 0: return False  
    if board[piece_x][piece_y] == 0: return False  

    if abs(attacker) == KING:
        cpy = copy.deepcopy(board)
        cpy[attacker_x][attacker_y], cpy[piece_x][piece_y] = 0, attacker
        if isCheck(cpy):
            return False

    if abs(attacker) == POWN:
        # en passant: my favorite move
        if (lastmove and 
            board[lastmove[1][0]][lastmove[1][1]] == -attacker and 
            lastmove[1][0] == lastmove[0][0] + 2 and 
            lastmove[1][0] == attacker_x and
            (lastmove[1][1] == attacker_y + 1 or lastmove[1][1] == attacker_y - 1)):
            if piece_x == lastmove[0][0] and piece_y == lastmove[0][1]:
                return True
        if piece_x<attacker_x and attacker<0:
            return False
        if piece_x>attacker_x and attacker>0:
            return False

    for d in directionGrid[abs(attacker)]:
        if abs(attacker) in [KNIGHT, KING]:
            x__ = attacker_x + d[0]
            y__ = attacker_y + d[1]
            
            if 0 <= x__ <= 7 and 0 <= y__ <= 7:
                if x__ == piece_x and y__ == piece_y:
                    if abs(board[piece_x][piece_y]) == KING: return 2

                    return True
        
        elif abs(attacker) == POWN:
            direction = -int(attacker/abs(attacker))
            x__ = attacker_x + direction * d[0]
            y__ = attacker_y + d[1]
            
            if 0 <= x__ <= 7 and 0 <= y__ <= 7:
                if x__ == piece_x and y__ == piece_y:
                    if d[1] != 0 and board[piece_x][piece_y] != 0:
                        if abs(board[piece_x][piece_y]) == KING: return 2
                        return True
        
        else:
            for i in range(1, 8):
                x__ = attacker_x + i * d[0]
                y__ = attacker_y + i * d[1]
                
                if x__ < 0 or y__ < 0 or x__ > 7 or y__ > 7:
                    break  
                
                if x__ == piece_x and y__ == piece_y:
                    if abs(board[piece_x][piece_y]) == KING: return 2
                    return True  
                
                if board[x__][y__] != 0:
                    break  
    
    return False

# def canTake(board, attacker_x,attacker_y , piece_x,piece_y,lastmove):
#     attacker = board[attacker_x][attacker_y]

#     if abs(attacker) == KING:
#         cpy = copy.deepcopy(board)
#         cpy[attacker_x][attacker_y],cpy[piece_x][piece_y] = 0, attacker
#         if isCheck(cpy):
#             return False
#     if abs(board[piece_x][piece_y]) == KING: return False
#     if abs(attacker) == PLACEHOLDER:return False
#     if attacker*board[piece_x][piece_y]>0: return False
#     if abs(attacker) == POWN :

        
        
#         if board[lastmove[1][0]][lastmove[1][1]] == -attacker and lastmove[1][0] == lastmove[0][0]+2 and (lastmove[1][0] == attacker_x and(lastmove[1][1] == attacker_y + 1 or lastmove[1][1] == attacker_y - 1)):
#             if piece_x == lastmove[0][0] and piece_y==lastmove[0][1]: return True
#     if board[piece_x][piece_y ]== 0:return False
#     for d in directionGrid[abs(attacker)]:
#         Isblock = False
#         for i in range(8):
#             x__,y__ = attacker_x+i*d[0], attacker_y+i*d[1]
            
#             if abs(attacker) in [KNIGHT, KING]:
#                 i = 1
#                 x__,y__ = attacker_x+i*d[0], attacker_y+i*d[1]
                
#             if abs(attacker) ==POWN:
#                 i=1
#                 x__,y__ = attacker_x-int(attacker/abs(attacker))*i*d[0], attacker_y+i*d[1]


                
#             if x__<0 or y__<0 or x__>7 or y__>7: continue
#             if (board[x__][y__] != 0 and x__!=piece_x and y__!=piece_y ) :
                
#                 Isblock = True
#                 break
#             if (x__==piece_x and y__==piece_y):
                
#                 x,y = piece_x,piece_y
#                 if abs(attacker) in [KNIGHT, KING,POWN]:
#                     return True 
#                 while x != attacker_x:
#                     x -= d[0]
#                     y -= d[1]
#                     if board[x][y] !=0:
#                         return False
#                 return not Isblock



#             if abs(attacker) in [KING,KNIGHT, POWN]:
#                 break
#     return False

def getAllpiece(board):
    pieces_info = {
        WHITE:{
            KING:[],
            POWN:[],
            BISHOP:[],
            KNIGHT:[],
            ROOK:[],
            QUEEN:[],
            PLACEHOLDER:[]
        },
        BLACK:{
            KING:[],
            POWN:[],
            BISHOP:[],
            KNIGHT:[],
            ROOK:[],
            QUEEN:[],
            PLACEHOLDER:[]
        }
    }

    for x,i in enumerate(board):
        for y,j in enumerate(i):
            if j == PLACEHOLDER:board[x][y] = 0
            if j!=0: pieces_info[j/abs(j)][abs(j)].append(( x,y))
    
    
    return pieces_info

def isCheck(board):
    
    allpiece = getAllpiece(board)
    color = (WHITE,BLACK)
    for i in color:
        if not allpiece[i][KING]:
            return False
        
        king_x,king_y = allpiece[i][KING][0]

        for k,v in allpiece[-i].items():
            for x,y in v:
                if canTake(board, x,y, king_x, king_y,((-1,-1),(-1,-1))) == 2:
                    return True
    return False

def isCheck_(board,clr):
    
    allpiece = getAllpiece(board)
    color = (WHITE,BLACK)
    if not allpiece[clr][KING]:
        return False
    
    king_x,king_y = allpiece[clr][KING][0]

    for k,v in allpiece[-clr].items():
        for x,y in v:
            if canTake(board, x,y, king_x, king_y,((-1,-1),(-1,-1))) == 2:
                return True
    return False

def canDoThat(board_, x_,y_,x,y,lastmove):
    if board_[x_][y_] == 0:return False
    clr = abs(board_[x_][y_])/board_[x_][y_]
    if isCheck_(board_,clr):
        cpy=copy.deepcopy(board_)
        cpy[x_][y_], cpy[x][y] = 0, cpy[x_][y_]
        if isCheck_(cpy,clr):
            return False
    board = copy.deepcopy(board_)
    if x<0 or y<0 or x>7 or y>7 or board[x_][y_] == 0: return False
    if board[x][y]==0 and abs(board[x_][y_]) != POWN:
        board[x][y] = -(board[x_][y_]/abs(board[x_][y_]))*PLACEHOLDER
    att = board[x_][y_]
    if abs(att) == PLACEHOLDER:return False
    
    if abs(att) == POWN :
        
        if x_ == 1  :
            if (x == x_+1 or x==x_+2) and y==y_:
                if board[x][y_] ==0 and sum((board[x_+1][y],board[x_+2][y] )) == 0:return True
            if x<x_:return False
        if  x_ == 6 :
            if ((x == x_-1 or x==x_-2) ) and y==y_: 
                if board[x][y_] ==0 and sum((board[x_-1][y],board[x_-2][y]) ) == 0:return True
            if x<x_:return False
    
    if canTake(board, x_,y_,x,y, lastmove) :
        
        
        return True
    
    return False



def isValidMove(board,x_,y_,x,y,lastmove):
    
    cpy_board = copy.deepcopy(board)
   
    if x<0 or y<0 or x>7 or y>7 or board[x_][y_] == 0: return False
    cpy_board[x_][y_] = 0
    cpy_board[x][y] = board[x_][y_]
    
    
    
    if board[x_][y_]==0 :
        return False
    
    
    allpiece = getAllpiece(cpy_board)
    if len(allpiece[BLACK][KING]) != len(allpiece[WHITE][KING]) and len(allpiece[WHITE][KING])==0:return False
    
    
    if isCheck_(board, -int(board[lastmove[1][0]][lastmove[1][1]]/abs(board[lastmove[1][0]][lastmove[1][1]]))):
        cpy=copy.deepcopy(board)
        cpy[x_][y_], cpy[x][y] = 0, cpy[x_][y_]
        if isCheck(cpy):
            return False
    # if isCheck(cpy_board): return False
    
    if canDoThat(board, x_,y_,x,y, lastmove): 
        
        return True
    
    return False

def getAllValidMove(board, color,lastmove):
    allMoveforplayer = getAllpiece(board)[color]
    movelist = []
    for k,v_ in allMoveforplayer.items():
        for v in v_: 
            for i in range(8):
                for j in range(8):
                    if i==v[0] and j==v[1]:continue
                    if isValidMove(board,v[0], v[1],i,j,lastmove):
                        movelist.append((v,(i,j)))
    
    
    return movelist


class Player:
    def __init__(self, model):
        self._clr = NULL
        self.model = model
        self.superlasmoves_grid = []
        self.lastmoves = [[0,0,0,0,0],
                          [0,0,0,0,0],
                          [0,0,0,0,0]
                          ]
    def setLastmoves(self, move):
        for i in range(len(self.lastmoves)):
            if not self.lastmoves[i][0] :
                self.lastmoves[i] = move.copy()
                return
        for i in range(len(self.lastmoves)-1):
            self.lastmoves[i] = self.lastmoves[i+1]
        self.lastmoves[-1] = move.copy()
        return
     

    def play(self,board,lastmove, move_counter):
        self.setLastmoves([board[lastmove[1][0]][lastmove[1][1]]]+[j for i in lastmove for j in i])
        punition = 0
        if self._clr != NULL:
            valid_moves = getAllValidMove(board, self._clr,lastmove)            
            eval_score = []
            for i in valid_moves:
                board_ = copy.deepcopy(board)
                board_[i[1][0]][i[1][1]], board_[i[0][0]][i[0][1]] = board_[i[0][0]][i[0][1]],0
                self.superlasmoves_grid.append(self.lastmoves)
                punition = move_counter(board_)*0.12
                # print(punition)
                eval_score.append((i, (1-punition)*self.model.eval_(board_,self.lastmoves)))
            # print(eval_score)
            best_move = max(eval_score, key=lambda v:v[1]) if self._clr>0 else max(eval_score, key=lambda v:-v[1])
            out = *best_move[0][0],*best_move[0][1]
            self.setLastmoves([board[out[2]][out[3]]]+list(out))
            return out
        return -1,-1,-1,-1
class HumanPlayer:
    def __init__(self):
        self._clr = NULL
    def play(self,board, lastmove,_):
        try:
            if self._clr == NULL: return -1,-1,-1,-1
            x_ = int(input("enter x_: "))
            y_ = int(input("enter y_: "))
            x  = int(input("enter x : "))
            y  = int(input("enter y : "))
            while not isValidMove(board, x_,y_,x,y,lastmove):
                print("Enter valid move: ")
                x_ = int(input("enter x_: "))
                y_ = int(input("enter y_: "))
                x  = int(input("enter x : "))
                y  = int(input("enter y : "))
            return x_,y_,x,y
        except:
            self.play(board, lastmove,_)
    

class chessGame:
    def __init__(self, player1, player2, isRandom=True):
        self.board = copy.deepcopy(INITCHESSGAME)
        self.lastmove = ((-1,-1),(-1,-1))
        self.gameHistory = []
        self.boardHistory = []  
        self.WHITE = player1
        self.BLACK = player2
        self.W,self.B = "P1","P2"

        if isRandom:
            if random.randint(0,1) == 1:
                self.WHITE = player2
                self.BLACK = player1
                self.W,self.B = "P2","P1"

        self.move_counter = 0  
        self.last_capture_or_pawn_move = 0  
        self.positions = []

        
        self.WHITE._clr = WHITE
        self.BLACK._clr = BLACK
        self.currentPlayer = self.WHITE
        
        
        self.boardHistory.append(self.board_to_str())
    def board_to_str(self, board=None):
        result = ""
        board = board if board else self.board
        for row in board:
            for cell in row:
                result += str(cell) + ","
        return result
    
    def update_lastmove(self):
        self.lastmove = self.gameHistory[-1]
    
    def togglePlayer(self):
        self.currentPlayer = self.WHITE if self.currentPlayer==self.BLACK else self.BLACK
    def countrep(self, current_state=None):
        current_state = self.board_to_str()  if not  current_state else self.board_to_str(current_state)
        count = 0
        for state in self.boardHistory:
            if state == current_state:
                count += 1
        return count
    def is_threefold_repetition(self):
        current_state = self.board_to_str()
        count = self.countrep()
        if count >= 3:
            return True
        return False
    
    def is_fifty_move_rule(self):
        
        return self.move_counter >= 100  
    
    def isEnded(self):
        valid_moves = getAllpiece(self.board)
        black_move, white_move = getAllValidMove(self.board, BLACK, self.lastmove), getAllValidMove(self.board, WHITE, self.lastmove)
        # print(f"white:{len(white_move)}---black:{len(black_move)}")
        if self.is_threefold_repetition():
            return NULL
        
        
        if self.is_fifty_move_rule():
            return NULL
        if len(black_move) == 0:
            
            if isCheck_(self.board, BLACK):  
                return WHITE  
            return NULL  

        
        if len(white_move) == 0: 
                
            if isCheck_(self.board,WHITE):  
                return BLACK  
            return NULL  
        
        
        
            
        return False
    
    def printGame(self):
        for i in self.board:
            print("| ", end="")
            for j in i:
                if j==0:
                    print("   | ", end="")
                else:
                    flag = "b" if j < 0 else "w"
                    if abs(j) == POWN: print(f"P{flag}", end=" | ")
                    elif abs(j) == KNIGHT: print(f"C{flag}", end=" | ")
                    elif abs(j) == BISHOP: print(f"F{flag}", end=" | ")
                    elif abs(j) == ROOK: print(f"T{flag}", end=" | ")
                    elif abs(j) == QUEEN: print(f"D{flag}", end=" | ")
                    elif abs(j) == KING : print(f"R{flag}", end=" | ")
            print("\n")

    def play(self):
        self.printGame()
        while self.isEnded() == False:
            # self.printGame()
            x_, y_, x, y = self.currentPlayer.play(self.board, self.lastmove, self.countrep)
            
            if isValidMove(self.board, x_, y_, x, y, self.lastmove):
                
                is_capture = self.board[x][y] != 0
                is_pawn_move = abs(self.board[x_][y_]) == POWN
                
                
                self.board[x][y] = self.board[x_][y_]
                self.board[x_][y_] = 0
                
                
                self.gameHistory.append(((x_, y_), (x, y)))
                self.update_lastmove()
                
                
                if is_capture or is_pawn_move:
                    self.move_counter = 0  
                    self.last_capture_or_pawn_move = len(self.gameHistory)
                else:
                    self.move_counter += 1
                
                self.positions.append(self.board)
                self.boardHistory.append(self.board_to_str())
                
                
                self.togglePlayer()
                self.printGame()
        if self.isEnded() == 2:
            print("That's a draw")
            return self.isEnded()  
        if self.isEnded() == WHITE:
            print(self.W)
            return self.isEnded()
        # print(self.B)
        return self.isEnded()  

        


def train(engine, positions, lastmoves, result,clr, lr=0.0001):
    engine.train()
    optimizer = optim.SGD(engine.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    rewards = []
    gamma = 0.99    
    reward = 0 if result == NULL else (1 if result == clr else -1)
    
    for i in range(len(positions)-1, -1, -1):
        rewards.insert(0, reward)
        reward = reward * gamma
    
    for pos, lastmove,reward in zip(positions, lastmoves,rewards):
        # print("hello")
        optimizer.zero_grad()
        out = engine.eval_(pos,lastmove)
        loss = criterion(out, torch.tensor([reward], dtype=torch.float32))
        loss.backward()
        optimizer.step()
for _ in range(10):
    game = chessGame(Player(engine1), HumanPlayer())
    game.play()
    if isinstance(game.WHITE, Player):
        train(game.WHITE.model,game.positions,game.WHITE.superlasmoves_grid,game.isEnded() if game.isEnded() != 2 else 0,game.WHITE._clr, 0.01)   
    else:
        train(game.BLACK.model,game.positions,game.WHITE.superlasmoves_grid,game.isEnded() if game.isEnded() != 2 else 0,game.BLACK._clr, 0.01)   

game = chessGame(Player(engine1), Player(engine2))
game.play()



"""
Point du projet:
    Le concept m'as l'air viable et faisable.
    Actuellement, quand il joue contre lui meme,il repete la position et il y a match nulle ce qui est cohérent en fait, si il rejoue le meme coups, il evalue le coups et cà retourne la meme evaluation logique qu'il reprenne le coups qu'il pensait etre le meilleur contre l'adversaire.
    J'ai quelques solution ajouter un facteurs de decouragement contre la repetition.Ou encore lui faire prendre 3 coups en plus en fait peut etre que si le réseau sait qu'il a deja jouer le coups il ne le rejouera pas j'ajouterais à l'echiquier flatened une petite grille 5*3 flatened avec les ancien coups et qui par defaut serais pleine de 0.
    J'aime bien la derniere idée je vais l'implementer car elle restreint moins, elle autorise les repetitions mais juste si on ne peut pas faire mieux pour améliorer la position.
    Je pense revenir le finire dans quelques temps.
    Pour l'heur il va etre entrainer sur 1000 parties contre stockfish et on verra bien.

"""