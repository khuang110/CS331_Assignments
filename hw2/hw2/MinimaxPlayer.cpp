/*
 * MinimaxPlayer.cpp
 *
 *  Created on: Apr 17, 2015
 *      Author: wong
 */
#include <iostream>
#include <assert.h>
#include <climits>
#include "MinimaxPlayer.h"




MinimaxPlayer::MinimaxPlayer(char symb) :
	Player(symb)
{
}

MinimaxPlayer::~MinimaxPlayer()
{
}

void MinimaxPlayer::get_move(OthelloBoard* b, int& col, int& row)
{
	MiniMax(b, true, col, row, true);
}

MinimaxPlayer* MinimaxPlayer::clone()
{
	MinimaxPlayer *result = new MinimaxPlayer(symbol);
	return result;
}

auto
successor(OthelloBoard *b, char symb)->std::list<node_t*>
{
	int k, j;
	std::list<node_t*> s;

	for (int i = 0; i < 16; i++)
	{
		// Col, Row variables respectively.
		k = i % 4;
		j = i / 4;

		if (b->is_cell_empty(k, j) && b->is_legal_move(k, j, symb))
		{
			//node_t *tmp = new node_t;
			node_t *tmp = new node_t;
			tmp->col = k;
			tmp->row = j;
			tmp->val = -1;
			s.push_back(tmp);
		}
		
	}
		return s;
}

bool
check_final(OthelloBoard *b)
{
	const char p1 = b->get_p1_symbol();
	const char p2 = b->get_p2_symbol();
	if (b->has_legal_moves_remaining(p1) && b->has_legal_moves_remaining(p2))
		return false;
	return true;
}


int
get_diff(OthelloBoard *b)
{
	const char p1 = b->get_p1_symbol();
	const char p2 = b->get_p2_symbol();
	return b->count_score(p1) - b->count_score(p2);		// Changed to p2-p1 
}


auto
MinimaxPlayer::max(OthelloBoard* b, char symb, bool player_1, int &alpha, int &beta)->node_t*
{
	node_t* tmp_node = new node_t;
	if (check_final(b))
	{
		tmp_node->row = -1;
		tmp_node->col = -1;
		tmp_node->val = get_diff(b);
	}
	else
	{
		tmp_node->val = INT_MIN;
		std::list<node_t*> s = successor(b, symb);
		OthelloBoard* tmp = nullptr;

		for (auto succ : s)
		{
			tmp_node->col = succ->col;
			tmp_node->row = succ->row;
			tmp = new OthelloBoard(*b);
			tmp->play_move(succ->col, succ->row, symb);
			tmp_node = std::max(min(tmp, symb, player_1, alpha, beta), tmp_node);
			if (tmp_node->val >= beta)
			{
				//succ->val = get_diff(b);
				//return succ;
				//tmp_node->val = get_diff(b);
				//tmp_node->val = get_diff(tmp);
				return tmp_node;
			}
			alpha = std::max(alpha, tmp_node->val);
				//if (tmp_node->val > MEval->val)
				//{ 
				//	MEval->val = tmp_node->val;
				//	MEval->col = succ->col;
				//	MEval->row = succ->row;
				//}
		}
		delete tmp;
		destroy_list(s);
	}
	//tmp_node->val = get_diff(b);
	return tmp_node;
}


node_t*
MinimaxPlayer::min(OthelloBoard* b, char symb, bool player_1, int &alpha, int &beta)
{
	node_t* tmp_node = new node_t;
	if (check_final(b))
	{
		tmp_node->row = -1;
		tmp_node->col = -1;
		tmp_node->val = get_diff(b);
	}
	else
	{
		tmp_node->val = INT_MAX;
		std::list<node_t*> s = successor(b, symb);
		OthelloBoard* tmp = nullptr;

		for (auto succ : s)
		{
			tmp_node->col = succ->col;
			tmp_node->row = succ->row;
			tmp = new OthelloBoard(*b);
			tmp->play_move(succ->col, succ->row, symb);
			tmp_node = std::min(max(tmp, symb, player_1, alpha, beta), tmp_node);
			if (tmp_node->val <= alpha)
			{
				//succ->val = get_diff(b);
				//return succ;
				//tmp_node->val = get_diff(b);
				//tmp_node->val = get_diff(tmp);
				return tmp_node;
			}
			beta = std::min(beta, tmp_node->val);
			//if (tmp_node->val < MEval->val)
			//{
			//	MEval->val = tmp_node->val;
			//	MEval->col = succ->col;
			//	MEval->row = succ->row;
			//}
		}
		delete tmp;
		destroy_list(s);
	}
	//tmp_node->val = get_diff(b);
	return tmp_node;
}


void
MinimaxPlayer::MiniMax(OthelloBoard* b, bool is_max, int &col, int &row, bool player_1)
{
	char symb;
	//auto *eval = new node_t;
	node_t eval;
	int alpha = INT_MIN,
		beta = INT_MAX;
	eval.val = -1;
	eval.col = -1;
	eval.row = -1;
	
	if (check_final(b))
	{
		symb = (player_1) ? b->get_p1_symbol() : b->get_p2_symbol();
		std::list<node_t*> s = successor(b, symb);
		for (auto succ : s)
		{
			col = succ->col;
			row = succ->row;
		}
	}
	else
	{
		// Is player Maximizing?
		if (is_max)
		{
			symb = (player_1) ? b->get_p1_symbol() : b->get_p2_symbol();
			//while ()
			eval = *max(b, symb, player_1, alpha, beta);
		}
		else
		{
			symb = (player_1) ? b->get_p1_symbol() : b->get_p2_symbol();
			eval = *min(b, symb, !player_1, alpha, beta);
		}
		col = eval.col;
		row = eval.row;
	}
}

//auto
//MinimaxPlayer::max(OthelloBoard *b, node_t *MaxEval, node_t *eval) -> node_ptr_t
//{
//	node_ptr_t max_node = new node_t;
//	max_node->val = MaxEval->val;
//	max_node->row = MaxEval->row;
//	max_node->col = MaxEval->col;
//	if (MaxEval->val < eval->val)
//	{
//		max_node->val = get_diff(b);
//		max_node->row = eval->row;
//		max_node->col = eval->col;
//	}
//	return max_node;
//}
//
//
//auto
//MinimaxPlayer::min(OthelloBoard *b, node_t *MinEval, node_t *eval) -> node_ptr_t
//{
//	node_ptr_t min_node = new node_t;
//	min_node->val = MinEval->val;
//	min_node->row = MinEval->row;
//	min_node->col = MinEval->col;
//	if (MinEval->val > eval->val)
//	{
//		min_node->val = get_diff(b);
//		min_node->col = eval->col;
//		min_node->row = eval->row;
//	}
//	return min_node;
//}
//
//auto
//MinimaxPlayer::MiniMax(OthelloBoard *b, int depth, node_ptr_t pos, bool is_max, bool player_1) -> node_ptr_t
//{
//	if (depth == 0 || check_final(b))
//	{
//		//pos.val = get_diff(b);
//		//pos.col = -1;
//		//pos.row = -1;
//		return pos;
//	}
//	char symb;
//	std::list<node_ptr_t> s;
//	//auto *eval = new node_t;
//	node_ptr_t eval;
//
//	// Is player Maximizing?
//	if (is_max)
//	{
//		symb = (player_1) ? b->get_p1_symbol() : b->get_p2_symbol();
//		s = successor(b, symb);
//		//node_t *maxEval = nullptr;
//		node_ptr_t maxEval = new node_t;
//		for (auto succ : s)
//		{
//			OthelloBoard* tmp = new OthelloBoard(*b);
//			tmp->play_move(succ->col, succ->row, symb);
//			//std::list<node_ptr_t> s2 = successor(tmp, symb);
//			eval = MiniMax(tmp, depth - 1, succ, false, !player_1);
//			//maxEval = new node_t;
//			maxEval->val = INT_MIN;
//			maxEval = max(tmp, maxEval, succ);
//		}
//		return maxEval;
//	}
//	else
//	{
//		symb = (player_1) ? b->get_p1_symbol() : b->get_p2_symbol();
//		s = successor(b, symb);
//		//node_t *minEval = nullptr;
//		auto minEval = new node_t;
//		for (auto const succ : s)
//		{
//			OthelloBoard *tmp = new OthelloBoard(*b);
//			tmp->play_move(succ->col, succ->row, symb);
//			//std::list<node_ptr_t> s2 = successor(tmp, symb);
//			eval = MiniMax(tmp, depth - 1, succ, true, !player_1);
//			//minEval = new node_t;
//			minEval->val = INT_MAX;
//			minEval = min(tmp, minEval, succ);
//			delete tmp;
//		}
//		destroy_list(s);
//		return minEval;
//	}
//}
