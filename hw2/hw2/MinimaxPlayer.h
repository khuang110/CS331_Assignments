/*
 * MinimaxPlayer.h
 *
 *  Created on: Apr 17, 2015
 *      Author: wong
 */

#ifndef MINIMAXPLAYER_H
#define MINIMAXPLAYER_H

#include <list>

#include "OthelloBoard.h"
#include "Player.h"
#include <vector>
#include <memory>

typedef struct Node
{
	int row,
	    col,
	    val;
} node_t;

inline bool operator<(const node_t &node, const node_t &other) { return node.val < other.val; }
inline bool operator>(const node_t &node, const node_t &other) { return node.val > other.val; }
inline bool operator==(const node_t &node, const node_t &other) { return node.val == other.val; }

typedef node_t *node_ptr_t;

/**
 * This class represents an AI player that uses the Minimax algorithm to play the game
 * intelligently.
 */
class MinimaxPlayer : public Player
{
public:

	/**
	 * @param symb This is the symbol for the minimax player's pieces
	 */
	MinimaxPlayer(char symb);

	/**
	 * Destructor
	 */
	virtual ~MinimaxPlayer();

	/**
	 * @param b The board object for the current state of the board
	 * @param col Holds the return value for the column of the move
	 * @param row Holds the return value for the row of the move
	 */
	void get_move(OthelloBoard *b, int &col, int &row);

	auto max(OthelloBoard *b, char symb, bool player_1, int &alpha, int &beta) -> node_t*;
	//auto max(OthelloBoard* b, node_t *MaxEval, node_t *eval) ->node_ptr_t;
	//auto min(OthelloBoard *b, char symb, node_t &MinEval, node_t eval)->node_t*;
	node_t* min(OthelloBoard *b, char symb, bool player_1, int &alpha, int &eta);
	//auto min(OthelloBoard* b, node_t *MinEval, node_t *eval) ->node_ptr_t;

	//auto MiniMax(OthelloBoard *b, int depth, node_ptr_t pos, bool is_max, bool player_1) ->node_ptr_t;
	void MiniMax(OthelloBoard *b, bool is_max, int &col, int &row, bool player_1);

	/**
	 * @return A copy of the MinimaxPlayer object
	 * This is a virtual copy constructor
	 */
	MinimaxPlayer* clone();

private:
	std::vector<std::list<node_t*>*> m;
};

/**
 *  @param  b: The board containing current state of game
 */
auto successor(OthelloBoard *b, char symb) -> std::list<node_t*>;

/***
 * Returns true both players have moves left
 * @param b: Othello board
 * @param symb: 
 */
bool check_final(OthelloBoard *b);


int get_diff(OthelloBoard *b);

template <typename T>
void destroy_list(std::list<T*> &v)
{
	while (!v.empty())
	{
		delete v.back();
		v.pop_back();
	}
}


#endif
