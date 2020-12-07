# lifelong-planing-pacman

Astar baseline 

Small maze:
python pacman.py -l smallMaze -z .5 -p SearchAgent -a fn=astarb,heuristic=manhattanHeuristic

Medium maze:
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astarb,heuristic=manhattanHeuristic

Big maze:
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astarb,heuristic=manhattanHeuristic		


Lifelong Astar

Small maze: 
python pacman.py -l smallMaze -z .5 -p SearchAgent -a fn=lstar,heuristic=manhattanHeuristic

Medium maze:
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=lstar,heuristic=manhattanHeuristic

Big maze:
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=lstar,heuristic=manhattanHeuristic

Dynamic Astar

small:
python pacman.py -l smallMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic

Medium:
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic

Big maze:
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic

