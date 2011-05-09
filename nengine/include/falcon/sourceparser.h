/* 
 * File:   sourceparser.h
 * Author: gian
 *
 * Created on 11 aprile 2011, 0.40
 */

#ifndef SOURCEPARSER_H
#define	SOURCEPARSER_H

/*
   FALCON - The Falcon Programming Language.
   FILE: sourceparser.cpp

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Apr 2011 00:04:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/parser/parser.h>

namespace Falcon {
class SynTree;

/** Class reading a Falcon script source.
 */
class FALCON_DYN_CLASS SourceParser: public Parsing::Parser
{
public:
   SourceParser();
   bool parse();

   //===============================================
   // Terminal tokens
   //
   Parsing::Terminal T_Plus;
   Parsing::Terminal T_Times;
   Parsing::Terminal T_Divide;
   Parsing::Terminal T_Minus;
   Parsing::Terminal T_UnaryMinus;
   Parsing::Terminal T_Modulo;
   Parsing::Terminal T_Power;
   Parsing::Terminal T_EqSign;
   Parsing::Terminal T_Arrow;
   Parsing::Terminal T_DblEq;
   Parsing::Terminal T_NotEq;
   Parsing::Terminal T_Less;
   Parsing::Terminal T_Greater;
   Parsing::Terminal T_LE;
   Parsing::Terminal T_GE;

   Parsing::Terminal T_Openpar;
   Parsing::Terminal T_Closepar;
   Parsing::Terminal T_OpenSquare;
   Parsing::Terminal T_CloseSquare;
   Parsing::Terminal T_OpenGraph;
   Parsing::Terminal T_CloseGraph;

   Parsing::Terminal T_Dot;
   Parsing::Terminal T_DotPar;
   Parsing::Terminal T_DotSquare;
   Parsing::Terminal T_Comma;
   Parsing::Terminal T_Colon;
   Parsing::Terminal T_Cut;

   Parsing::Terminal T_as;
   Parsing::Terminal T_eq;
   Parsing::Terminal T_if;
   Parsing::Terminal T_in;
   Parsing::Terminal T_or;
   Parsing::Terminal T_to;
   
   Parsing::Terminal T_and;
   Parsing::Terminal T_def;
   Parsing::Terminal T_end;
   Parsing::Terminal T_for;
   Parsing::Terminal T_not;
   Parsing::Terminal T_nil;
   Parsing::Terminal T_try;

   Parsing::Terminal T_elif;
   Parsing::Terminal T_else;

   Parsing::Terminal T_while;
   Parsing::Terminal T_rule;

   Parsing::Terminal T_function;
   Parsing::Terminal T_return;

   //================================================
   // Statements
   //

   Parsing::NonTerminal S_Autoexpr;
   Parsing::Rule r_line_autoexpr;

   Parsing::NonTerminal S_If;
   Parsing::Rule r_if;
   Parsing::Rule r_if_short;

   Parsing::NonTerminal S_Elif;
   Parsing::Rule r_elif;
   Parsing::NonTerminal S_Else;
   Parsing::Rule r_else;

   Parsing::NonTerminal S_While;
   Parsing::Rule r_while;
   Parsing::Rule r_while_short;

   Parsing::NonTerminal S_Rule;
   Parsing::Rule r_rule;
   
   Parsing::NonTerminal S_Cut;
   Parsing::Rule r_cut;

   Parsing::NonTerminal S_End;
   Parsing::Rule r_end;
   Parsing::Rule r_end_rich;
   
   //================================================
   // Expression
   //
   Parsing::NonTerminal Expr;

   Parsing::Rule r_Expr_assign;
   Parsing::Rule r_Expr_equal;
   Parsing::Rule r_Expr_diff;
   Parsing::Rule r_Expr_less;
   Parsing::Rule r_Expr_greater;
   Parsing::Rule r_Expr_le;
   Parsing::Rule r_Expr_ge;
   Parsing::Rule r_Expr_eeq;

   Parsing::Rule r_Expr_call;
   Parsing::Rule r_Expr_index;
   Parsing::Rule r_Expr_star_index;
   Parsing::Rule r_Expr_empty_dict;
   Parsing::Rule r_Expr_array_decl;
   Parsing::Rule r_Expr_empty_dict2;
   Parsing::Rule r_Expr_array_decl2;
   Parsing::Rule r_Expr_dot;
   Parsing::Rule r_Expr_plus;
   Parsing::Rule r_Expr_minus;
   Parsing::Rule r_Expr_pars;
   Parsing::Rule r_Expr_pars2;
   Parsing::Rule r_Expr_times;
   Parsing::Rule r_Expr_div;
   Parsing::Rule r_Expr_pow;
   Parsing::Rule r_Expr_neg;
   Parsing::Rule r_Expr_neg2;
   Parsing::Rule r_Expr_Atom;

   Parsing::Rule r_Expr_function;

   //================================================
   // Function
   //

   Parsing::NonTerminal S_Function;
   Parsing::Rule r_function;

   Parsing::NonTerminal S_Return;
   Parsing::Rule r_return;


   //================================================
   // Atom
   //
   Parsing::NonTerminal Atom;
   Parsing::Rule r_Atom_Int;
   Parsing::Rule r_Atom_Float;
   Parsing::Rule r_Atom_Name;
   Parsing::Rule r_Atom_String;
   Parsing::Rule r_Atom_Nil;

   //================================================
   // Expression lists
   //
   Parsing::NonTerminal ListExpr;
   Parsing::Rule r_ListExpr_next;
   Parsing::Rule r_ListExpr_first;
   Parsing::Rule r_ListExpr_empty;

   Parsing::NonTerminal ListExprOrPairs;
   Parsing::Rule r_ListExprOrPairs_next_pair;
   Parsing::Rule r_ListExprOrPairs_next;
   Parsing::Rule r_ListExprOrPairs_first_pair;
   Parsing::Rule r_ListExprOrPairs_first;
   Parsing::Rule r_ListExprOrPairs_empty;

   Parsing::NonTerminal SeqExprOrPairs;
   Parsing::Rule r_SeqExprOrPairs_next_pair_cm;
   Parsing::Rule r_SeqExprOrPairs_next_pair;
   Parsing::Rule r_SeqExprOrPairs_next_cm;
   Parsing::Rule r_SeqExprOrPairs_next;
   Parsing::Rule r_SeqExprOrPairs_first_pair;
   Parsing::Rule r_SeqExprOrPairs_first;
   Parsing::Rule r_SeqExprOrPairs_empty;

   //================================================
   // Symbol list
   //

   Parsing::NonTerminal ListSymbol;
   Parsing::Rule r_ListSymbol_next;
   Parsing::Rule r_ListSymbol_first;
   Parsing::Rule r_ListSymbol_empty;

   //================================================
   // States
   //

   Parsing::State s_Main;
};

}

#endif	/* SOURCEPARSER_H */

/* end of sourceparser.h */
