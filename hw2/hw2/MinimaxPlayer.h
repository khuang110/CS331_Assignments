/*
 * MinimaxPlayer.h
 *
 *  Created on: Apr 17, 2015
 *      Author: wong
 */

#ifndef MINIMAXPLAYER_H
#define MINIMAXPLAYER_H

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

typedef std::unique_ptr<node_t> node_ptr_t;

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

	//auto max(OthelloBoard *b, char symb, node_t &MaxEval, node_t eval)->node_t*;
	auto max(OthelloBoard* b, node_t *MaxEval, node_t *eval) ->node_ptr_t;
	//auto min(OthelloBoard *b, char symb, node_t &MinEval, node_t eval)->node_t*;
	auto min(OthelloBoard* b, node_t *MinEval, node_t *eval) ->node_ptr_t;

	auto MiniMax(OthelloBoard *b, int depth, node_t &pos, bool is_max, bool player_1) ->node_ptr_t;

	/**
	 * @return A copy of the MinimaxPlayer object
	 * This is a virtual copy constructor
	 */
	MinimaxPlayer* clone();

private:
};

/**
 *  @param  b: The board containing current state of game
 */
void successor(OthelloBoard *b, char symb, std::vector<node_ptr_t> &s);

/***
 * Returns true both players have moves left
 * @param b: Othello board
 * @param symb: 
 */
bool check_final(OthelloBoard *b);


int get_diff(OthelloBoard *b);

#endif
