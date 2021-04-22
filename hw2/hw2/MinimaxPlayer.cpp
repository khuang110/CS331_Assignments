/*
 * MinimaxPlayer.cpp
 *
 *  Created on: Apr 17, 2015
 *      Author: wong
 */
#include <iostream>
#include <assert.h>
#include "MinimaxPlayer.h"

#include <list>


MinimaxPlayer::MinimaxPlayer(char symb) :
	Player(symb)
{
}

MinimaxPlayer::~MinimaxPlayer()
{
}

void MinimaxPlayer::get_move(OthelloBoard *b, int &col, int &row)
{
	node_ptr_t pos;
	node_t* pt;
	//pos.val = get_diff(b);
	//pos.col = col;
	//pos.row = row;
	pt = MiniMax(b, 3, *pos, true, true);
	int i;
}

MinimaxPlayer* MinimaxPlayer::clone()
{
	MinimaxPlayer *result = new MinimaxPlayer(symbol);
	return result;
}

void
successor(OthelloBoard *b, char symb, std::list<node_ptr_t> &s)
{
	int k, j;

	for (int i = 0; i < 16; i++)
	{
		// Col, Row variables respectively.
		k = i % 4;
		j = i / 4;

		if (b->is_cell_empty(k, j) && b->is_legal_move(k, j, symb))
		{
			//node_t *tmp = new node_t;
			node_ptr_t tmp(new node_t);
			tmp->col = k;
			tmp->row = j;
			tmp->val = INT_MIN;
			s.push_back(std::move(tmp));
		}
	}
}

bool
check_final(OthelloBoard *b)
{
	/*
	 *  Maybe add swich if max p1-p2 else p2-p1....
	 */
	const char p1 = b->get_p1_symbol();
	const char p2 = b->get_p2_symbol();
	if (b->has_legal_moves_remaining(p1) && b->has_legal_moves_remaining(p2))
		return true;
	return false;
}


int
get_diff(OthelloBoard *b)
{
	const char p1 = b->get_p1_symbol();
	const char p2 = b->get_p2_symbol();
	return b->count_score(p1) - b->count_score(p2);
}


//node_t*
//MinimaxPlayer::max(OthelloBoard* b, char symb, node_t &MaxEval, node_t eval)
//{
//	OthelloBoard* tmp = nullptr;
//
//	if(check_final(b))
//	{
//		MaxEval.val = get_diff(b);
//		MaxEval.row = -1;
//		MaxEval.row = -1;
//	}
//	else
//	{
//		auto* tmp_node = new node_t;
//		if (tmp_node->val > MaxEval.val)
//		{
//			MaxEval.val = tmp_node->val;
//			MaxEval.row = tmp_node->row;
//			MaxEval.col = tmp_node->col;
//		}
//	}
//}

auto
MinimaxPlayer::max(OthelloBoard *b, node_t *MaxEval, node_t *eval) -> node_ptr_t
{
	node_ptr_t max_node;
	max_node->val = MaxEval->val;
	max_node->row = MaxEval->row;
	max_node->col = MaxEval->col;
	if (MaxEval->val < eval->val)
	{
		max_node->val = get_diff(b);
		max_node->row = eval->row;
		max_node->col = eval->col;
	}
	return max_node;
}


auto
MinimaxPlayer::min(OthelloBoard *b, node_t *MinEval, node_t *eval) -> node_ptr_t
{
	node_ptr_t min_node;
	min_node->val = MinEval->val;
	min_node->row = MinEval->row;
	min_node->col = MinEval->col;
	if (MinEval->val > eval->val)
	{
		min_node->val = get_diff(b);
		min_node->col = eval->col;
		min_node->row = eval->row;
	}
	return min_node;
}

auto
MinimaxPlayer::MiniMax(OthelloBoard *b, int depth, node_t &pos, bool is_max, bool player_1) -> node_ptr_t
{
	if (depth == 0 || check_final(b))
	{
		//pos.val = get_diff(b);
		//pos.col = -1;
		//pos.row = -1;
		return &pos;
	}
	char symb;
	std::list<node_ptr_t> s;
	//auto *eval = new node_t;
	node_ptr_t eval;

	// Is player Maximizing?
	if (is_max)
	{
		symb = (player_1) ? b->get_p1_symbol() : b->get_p2_symbol();
		successor(b, symb, s);
		//node_t *maxEval = nullptr;
		node_ptr_t maxEval;
		for (auto const &succ : s)
		{
			std::list<node_ptr_t> s2;
			OthelloBoard* tmp = new OthelloBoard(*b);
			tmp->play_move(succ->col, succ->row, symb);
			successor(tmp, symb, s2);
			eval = MiniMax(tmp, depth - 1, *succ, false, !player_1);
			//maxEval = new node_t;
			maxEval->val = INT_MIN;
			maxEval = max(tmp, &(*maxEval), &(*eval));
		}
		return maxEval;
	}
	else
	{
		symb = (player_1) ? b->get_p1_symbol() : b->get_p2_symbol();
		successor(b, symb, s);
		//node_t *minEval = nullptr;
		node_ptr_t minEval;
		for (auto const &succ : s)
		{
			std::list<node_ptr_t> s2;
			OthelloBoard *tmp = new OthelloBoard(*b);
			tmp->play_move(succ->col, succ->row, symb);
			successor(tmp, symb, s2);
			eval = MiniMax(tmp, depth - 1, *succ, true, !player_1);
			//minEval = new node_t;
			minEval->val = INT_MAX;
			minEval = min(tmp, &(*minEval), &(*eval));
		}
		return minEval;
	}
}
