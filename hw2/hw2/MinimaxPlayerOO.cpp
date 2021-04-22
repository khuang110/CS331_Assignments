/*
 * MinimaxPlayer.cpp
 *
 *  Created on: Apr 17, 2015
 *      Author: wong
 */
#include <iostream>
#include <assert.h>
#include "MinimaxPlayer.h"


using namespace std;

MinimaxPlayer::MinimaxPlayer(char symb) :
		Player(symb) {
}

MinimaxPlayer::~MinimaxPlayer() {

}


int MinimaxPlayer::Cal_unity(OthelloBoard* b){
	char player_1 = b->get_p1_symbol();
	char player_2 = b->get_p2_symbol();
	return (b->count_score(player_1)-b->count_score(player_2));
}


void MinimaxPlayer::Generate_Successor(OthelloBoard* b, char symb, int successor[16][2]){
	int p = 0;
	for(int i = 0; i < 16; i++){
		successor[i][0] = -1;
		successor[i][1] = -1;
	}
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			if(b->is_cell_empty(i,j)){
				if(b->is_legal_move(i,j,symb))
				{
					successor[p][0] = i;
					successor[p][1] = j;
					p++;
				}
			}
		}
	}
}


 int MinimaxPlayer::checkfinal(OthelloBoard* b){
	char player_1 = b->get_p1_symbol();
	char player_2 = b->get_p2_symbol();
	if(b->has_legal_moves_remaining(player_1) == false && b->has_legal_moves_remaining(player_2) == false){
		return -1;
	}

	return 0;
}


MinimaxPlayer_node MinimaxPlayer::Max_value(OthelloBoard* b, int istop){
	char player_1 = b->get_p1_symbol();

	int k = 0;
	int best_val = -999;
	MinimaxPlayer_node best_node;

	best_node.value = -9999;

	MinimaxPlayer_node tmp;
	int successors[16][2];
	OthelloBoard* newboard = NULL;

	if(checkfinal(b) == -1){
		best_node.value = Cal_unity(b);
		best_node.col = -1;
		best_node.row = -1;
		return best_node;
		cout << "-------------NO-----------" << endl;
	}
	else{
		Generate_Successor(b,player_1, successors);

		while(successors[k][0]!=-1 && k<16){
			newboard = new OthelloBoard(*b);
			newboard->play_move(successors[k][0], successors[k][1], player_1);
			tmp = Min_value(newboard);

			if(tmp.value > best_node.value){
				best_node.value = tmp.value;
				best_node.col = successors[k][0];
				best_node.row = successors[k][1];
			}
			k++;
		}

	}
	return best_node;
}


MinimaxPlayer_node MinimaxPlayer::Min_value(OthelloBoard* b, int istop){
	char player_2 = b->get_p2_symbol();
	int successors[16][2];
	int k = 0;

	MinimaxPlayer_node best_node;
	MinimaxPlayer_node tmp;
	OthelloBoard* newboard = NULL;

	best_node.value=99999;

	if(checkfinal(b) == -1){
		best_node.value = Cal_unity(b);
		best_node.col = -1;
		best_node.row = -1;
		return best_node;
	}
	else{
		Generate_Successor(b, player_2, successors);

		while(successors[k][0] != -1 && k < 16){
			newboard = new OthelloBoard(*b);
			newboard->play_move(successors[k][0], successors[k][1], player_2);
			tmp = Max_value(newboard);

			if(tmp.value < best_node.value){
				best_node.col = successors[k][0];
				best_node.row = successors[k][1];
				best_node.value = tmp.value;
			}
			k++;
		}
	}
	return best_node;
}


void MinimaxPlayer::Min_max(OthelloBoard* b, int &col, int &row){
	char player1 = b->get_p1_symbol();
	char player2 = b->get_p2_symbol();
	MinimaxPlayer_node res;

	if(symbol == player1){
		cout << "Select Max" << endl;
		res = Max_value(b);
	}
	else{
		cout << "Select Min" << endl;
		res = Min_value(b);
	}

	if(res.col != -1 && res.row != -1){
		col = res.col;
		row = res.row;
	}
}


void MinimaxPlayer::get_move(OthelloBoard* b, int &col, int &row) {
	Min_max(b, col, row);
}


MinimaxPlayer* MinimaxPlayer::clone() {
	MinimaxPlayer* result = new MinimaxPlayer(symbol);
	return result;
}