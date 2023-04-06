from antlr4 import *

from OCLMetric.AST_match.dist3.OclExpressionLexer import OclExpressionLexer 
from OCLMetric.AST_match.dist3.OclExpressionParser import OclExpressionParser
from OCLMetric.AST_match.dist3.OclExpressionVisitor import OclExpressionVisitor 

import queue

class node():
    def __init__(self,type,exp=None,children=[]):
        self.type=type
        self.exp=exp
        self.children=children
        self.tested=0
    def __eq__(self, other):
        if self.type==other.type:
            if self.exp==other.exp:
                if sorted(self.children)==sorted(other.children):
                    return True
        return False
    def addchild(self,child):
        self.children.append(child)

class Treematcher():
    ruleNames = None
    @staticmethod
    def getroot(statement):
        input_stream = InputStream(statement)
        lexer = OclExpressionLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OclExpressionParser(stream)
        root = parser.oclExp()
        Treematcher.ruleNames=parser.ruleNames
        return root

    # 给一个解析树点，返回一个node
    @staticmethod
    def getnode(cnode):
        sons=[]
        ttype= str(type(cnode))
        texp=cnode.toString(Treematcher.ruleNames,None)
        bg=texp.find('[')+1
        ed=texp.find(' ')
        texp=texp[bg:ed]
        for i in range(cnode.getChildCount()):
            sons.append(str(type(cnode.children[i])))
        result=node(type=ttype,exp=texp,children=sons)
        return result

    # 给定一个root节点，给出这棵树的所有子树列表
    # 队列维护BFS
    # 根节点则将其new一个node对象并将其儿子们的typestr之后放入列表
    @staticmethod
    def gettrees(root):
        trees=[]
        search_que=queue.Queue()
        search_que.put(root)
        # 保证入队的全是根节点
        while not search_que.empty():
            current_root = search_que.get()
            trees.append(Treematcher.getnode(current_root))
            for i in range(current_root.getChildCount()):
                temp = current_root.children[i]
                # 不是终结符
                if not isinstance(temp,tree.Tree.TerminalNodeImpl):
                    search_que.put(temp)
        return trees

    @staticmethod
    def match(clist,rlist):
        count_r=len(rlist)
        count_matched=0
        count1_matched=0
        for itemr in rlist:
            if itemr.tested==1:
                continue

            for itemc in clist:
                if itemc.tested==1:
                    continue

                if itemr == itemc:
                    count_matched+=1
                    itemr.tested=1
                    itemc.tested=1
                   # rlist.remove(itemr)
                   # clist.remove(itemc)
                    break
                    
        for item in rlist:
            item.tested = 0
        for item in clist:
            item.tested = 0
            
        for itemr in rlist:
            if itemr.tested==1:
                continue
            for itemc in clist:
                if itemc.tested==1:
                    continue            
                if itemr.exp == itemc.exp:
                    count1_matched+=1
                    itemr.tested=1
                    itemc.tested=1
                   # rlist.remove(itemr)
                   # clist.remove(itemc)
                    break
        #return (count_matched,count_r)
        
        return 0 if count_r==0 else count_matched*count1_matched/count_r/count_r


    @staticmethod
    def calculate(candidate,reference):
        can= Treematcher.getroot(candidate)
        ref= Treematcher.getroot(reference)
        #print(can.getText())
        #print(candidate)
        if can.getText()==candidate and ref.getText()==reference:
            clist = Treematcher.gettrees(can)
            rlist = Treematcher.gettrees(ref)
            return Treematcher.match(clist=clist,rlist=rlist)
        else:
            return 0