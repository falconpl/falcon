/*
   FALCON - The Falcon Programming Language.
   FILE: sourceparser.h

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Apr 2011 00:04:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef SOURCEPARSER_H
#define SOURCEPARSER_H

#include <falcon/setup.h>
#include <falcon/parser/parser.h>

namespace Falcon {
class SynTree;
class Error;

/** Class reading a Falcon script source.
 */
class FALCON_DYN_CLASS SourceParser: public Parsing::Parser
{
public:
   SourceParser();
   bool parse();

   virtual void onPushState( bool isPushedState );
   virtual void onPopState();

   /** Clears the source parser status. */
   virtual void reset();

   virtual void addError( int code, const String& uri, int l, int c, int ctx, const String& extra );
   virtual void addError( int code, const String& uri, int l, int c=0, int ctx=0  );   
   virtual void addError( Error* err );

   //===============================================
   // Terminal tokens
   //
   Parsing::Terminal T_Openpar;
   Parsing::Terminal T_Closepar;
   Parsing::Terminal T_OpenSquare;
   Parsing::Terminal T_CapPar;
   Parsing::Terminal T_CapSquare;
   Parsing::Terminal T_DotPar;
   Parsing::Terminal T_DotSquare;
   Parsing::Terminal T_CloseSquare;
   Parsing::Terminal T_OpenGraph;
   Parsing::Terminal T_OpenProto;
   Parsing::Terminal T_CloseGraph;
   Parsing::Terminal T_Dot;
   Parsing::Terminal T_Arrow;
   Parsing::Terminal T_AutoAdd;
   Parsing::Terminal T_AutoSub;
   Parsing::Terminal T_AutoTimes;
   Parsing::Terminal T_AutoDiv;
   Parsing::Terminal T_AutoMod;
   Parsing::Terminal T_AutoPow;
   Parsing::Terminal T_AutoRShift;
   Parsing::Terminal T_AutoLShift;
   Parsing::Terminal T_EEQ;
   
   Parsing::Terminal T_BAND;
   Parsing::Terminal T_BOR;
   Parsing::Terminal T_BXOR;
   Parsing::Terminal T_BNOT;
   Parsing::Terminal T_OOB;
   Parsing::Terminal T_DEOOB;
   Parsing::Terminal T_XOOB;
   Parsing::Terminal T_ISOOB;
   Parsing::Terminal T_UNQUOTE;
   Parsing::Terminal T_COMPOSE;
   Parsing::Terminal T_EVALRET;
   Parsing::Terminal T_EVALRET_EXEC;
   Parsing::Terminal T_EVALRET_DOUBT;
   Parsing::Terminal T_ETAARROW;
   
   Parsing::Terminal T_Comma;
   Parsing::Terminal T_QMark;
   Parsing::Terminal T_Tilde;
   Parsing::Terminal T_Bang;
   Parsing::Terminal T_Disjunct;

   Parsing::Terminal T_UnaryMinus;
   Parsing::Terminal T_Dollar;
   Parsing::Terminal T_Amper;
   Parsing::Terminal T_NumberSign;
   Parsing::Terminal T_At;
   Parsing::Terminal T_Power;
   Parsing::Terminal T_Times;
   Parsing::Terminal T_Divide;
   Parsing::Terminal T_Modulo;
   Parsing::Terminal T_RShift;
   Parsing::Terminal T_LShift;
   Parsing::Terminal T_Plus;
   Parsing::Terminal T_Minus;
   Parsing::Terminal T_PlusPlus;
   Parsing::Terminal T_MinusMinus;
   Parsing::Terminal T_DblEq;
   Parsing::Terminal T_NotEq;
   Parsing::Terminal T_Less;
   Parsing::Terminal T_Greater;
   Parsing::Terminal T_LE;
   Parsing::Terminal T_GE;
   Parsing::Terminal T_Colon;
   Parsing::Terminal T_EqSign;
   Parsing::Terminal T_EqSign2;
   Parsing::Terminal T_as;
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
   Parsing::Terminal T_catch;
   Parsing::Terminal T_notin;
   Parsing::Terminal T_finally;
   Parsing::Terminal T_raise;
   Parsing::Terminal T_fself;
   Parsing::Terminal T_elif;
   Parsing::Terminal T_else;
   Parsing::Terminal T_rule;
   Parsing::Terminal T_while;
   Parsing::Terminal T_function;
   Parsing::Terminal T_return;
   Parsing::Terminal T_class;
   Parsing::Terminal T_object;
   Parsing::Terminal T_init;

   Parsing::Terminal T_true;
   Parsing::Terminal T_false;
   Parsing::Terminal T_self;
   Parsing::Terminal T_from;
   Parsing::Terminal T_load;
   Parsing::Terminal T_export;
   Parsing::Terminal T_import;
   Parsing::Terminal T_namespace;
   Parsing::Terminal T_global;

   Parsing::Terminal T_forfirst;
   Parsing::Terminal T_formiddle;
   Parsing::Terminal T_forlast;
   Parsing::Terminal T_break;
   Parsing::Terminal T_continue;
   
   Parsing::Terminal T_switch;
   Parsing::Terminal T_case;
   Parsing::Terminal T_default;
   Parsing::Terminal T_select;
   Parsing::Terminal T_loop;
   Parsing::Terminal T_static;
   
   Parsing::Terminal T_RString;
   Parsing::Terminal T_IString;
   Parsing::Terminal T_MString;

   Parsing::Terminal T_provides;
   Parsing::Terminal T_DoubleColon;
   Parsing::Terminal T_ColonQMark;
   //================================================
   // Statements
   //

   Parsing::NonTerminal S_Statement;
   Parsing::Rule r_statement_fastprint;
   Parsing::Rule r_statement_if;

   Parsing::NonTerminal S_Attribute;
   Parsing::Rule r_attribute;

   Parsing::NonTerminal S_Autoexpr;
   Parsing::Rule r_line_autoexpr;
   Parsing::Rule r_assign_list;

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
   
   Parsing::NonTerminal S_Continue;
   Parsing::Rule r_continue;

   Parsing::NonTerminal S_Break;
   Parsing::Rule r_break;

   Parsing::NonTerminal S_For;
   Parsing::Rule r_for_to_step;
   Parsing::Rule r_for_to_step_short;
   Parsing::Rule r_for_to;
   Parsing::Rule r_for_to_short;
   Parsing::Rule r_for_in;
   Parsing::Rule r_for_in_short;
   
   Parsing::NonTerminal S_Forfirst;
   Parsing::Rule r_forfirst;
   Parsing::Rule r_forfirst_short;
   Parsing::NonTerminal S_Formiddle;
   Parsing::Rule r_formiddle;
   Parsing::Rule r_formiddle_short;
   Parsing::NonTerminal S_Forlast;
   Parsing::Rule r_forlast;
   Parsing::Rule r_forlast_short;
   
   Parsing::NonTerminal S_Switch;
   Parsing::Rule r_switch;
   
   Parsing::NonTerminal S_Select;
   Parsing::Rule r_select;
   
   Parsing::NonTerminal S_Case;
   Parsing::Rule r_case;
   Parsing::Rule r_case_short;
   
   Parsing::NonTerminal S_Default;
   Parsing::Rule r_default;
   Parsing::Rule r_default_short;
   
   Parsing::NonTerminal S_RuleOr;
   Parsing::Rule r_rule;
   Parsing::Rule r_rule_or;

   Parsing::NonTerminal S_Cut;
   Parsing::Rule r_cut_expr;
   Parsing::Rule r_cut;
   
   Parsing::NonTerminal S_Doubt;
   Parsing::Rule r_doubt;

   Parsing::NonTerminal S_End;
   Parsing::Rule r_end;
   Parsing::Rule r_end_rich;

   Parsing::NonTerminal S_SmallEnd;
   Parsing::Rule r_end_small;

   Parsing::NonTerminal S_EmptyLine;
   Parsing::Rule r_empty;

   Parsing::NonTerminal S_MultiAssign;
   Parsing::Rule r_Stmt_assign_list;
   
   Parsing::NonTerminal S_FastPrint;
   Parsing::Rule r_fastprint_alone;
   Parsing::Rule r_fastprint;
   Parsing::Rule r_fastprint_nl_alone;
   Parsing::Rule r_fastprint_nl;
   
   Parsing::NonTerminal PrintExpr;
   Parsing::Rule r_PrintExpr_next;
   Parsing::Rule r_PrintExpr_first;
   Parsing::Rule r_PrintExpr_empty;

   //================================================
   // Load, import and export
   //
   Parsing::NonTerminal S_Load;
   Parsing::Rule r_load_string;
   Parsing::Rule r_load_mod_spec;

   Parsing::NonTerminal ModSpec;
   Parsing::Rule r_modspec_next;
   Parsing::Rule r_modspec_first;
   Parsing::Rule r_modspec_first_self;
   Parsing::Rule r_modspec_first_dot;
   
   Parsing::NonTerminal S_Export;
   Parsing::Rule r_export_rule;
   
   Parsing::NonTerminal S_Global;
   Parsing::Rule r_global_rule;

   Parsing::NonTerminal S_Try;
   Parsing::Rule r_try_rule;

   Parsing::NonTerminal S_Catch;
   Parsing::Rule r_catch_base;
   Parsing::NonTerminal S_Finally;
   Parsing::Rule r_finally;

   Parsing::NonTerminal CatchSpec;
   Parsing::Rule r_catch_all;
   Parsing::Rule r_catch_in_var;
   Parsing::Rule r_catch_as_var;
   Parsing::Rule r_catch_number;
   Parsing::Rule r_catch_number_in_var;
   Parsing::Rule r_catch_number_as_var;
   Parsing::Rule r_catch_thing;
   Parsing::Rule r_catch_thing_in_var;
   Parsing::Rule r_catch_thing_as_var;

   Parsing::NonTerminal S_Raise;
   Parsing::Rule r_raise;
   
   Parsing::NonTerminal S_Import;
   Parsing::Rule r_import_rule;
   
   Parsing::NonTerminal ImportClause;   
   Parsing::Rule r_import_from_string_as;
   Parsing::Rule r_import_from_string_in;
   Parsing::Rule r_import_star_from_string_in;
   Parsing::Rule r_import_star_from_string;
   Parsing::Rule r_import_from_string;
   Parsing::Rule r_import_from_modspec_as;
   Parsing::Rule r_import_from_modspec_in;
   Parsing::Rule r_import_star_from_modspec_in;
   Parsing::Rule r_import_star_from_modspec;   
   Parsing::Rule r_import_from_modspec;   
   Parsing::Rule r_import_syms;
   
   Parsing::NonTerminal ImportSpec;
   Parsing::Rule r_ImportSpec_next;
   Parsing::Rule r_ImportSpec_attach_last;
   Parsing::Rule r_ImportSpec_attach_next;
   Parsing::Rule r_ImportSpec_first;
   Parsing::Rule r_ImportSpec_empty;
   
   Parsing::NonTerminal NameSpaceSpec;
   Parsing::Rule r_NameSpaceSpec_last;
   Parsing::Rule r_NameSpaceSpec_next;
   Parsing::Rule r_NameSpaceSpec_first;
   
   Parsing::NonTerminal S_Loop;
   Parsing::Rule r_loop_short;
   Parsing::Rule r_loop;

   Parsing::NonTerminal S_Namespace;
   Parsing::Rule r_NameSpace;
   
   //================================================
   // Expression
   //
   Parsing::NonTerminal Expr;

   Parsing::Rule r_Expr_assign;
   Parsing::Rule r_Expr_list;

   Parsing::Rule r_Expr_equal;
   Parsing::Rule r_Expr_diff;
   Parsing::Rule r_Expr_less;
   Parsing::Rule r_Expr_greater;
   Parsing::Rule r_Expr_le;
   Parsing::Rule r_Expr_ge;
   Parsing::Rule r_Expr_eeq;
   Parsing::Rule r_Expr_in;
   Parsing::Rule r_Expr_notin;

   Parsing::Rule r_Expr_call;
   Parsing::Rule r_Expr_summon;
   Parsing::Rule r_Expr_opt_summon;
   Parsing::Rule r_Expr_index;
   Parsing::Rule r_Expr_star_index;
   Parsing::Rule r_Expr_range_index3;
   Parsing::Rule r_Expr_range_index3open;
   Parsing::Rule r_Expr_range_index2;
   Parsing::Rule r_Expr_range_index1;
   Parsing::Rule r_Expr_range_index0;
   Parsing::Rule r_Expr_empty_dict;
   Parsing::Rule r_Expr_array_decl;
   Parsing::Rule r_Expr_empty_dict2;
   Parsing::Rule r_Expr_array_decl2;
   Parsing::Rule r_Expr_ref;
   Parsing::Rule r_Expr_amper;
   Parsing::Rule r_Expr_dot;
   Parsing::Rule r_Expr_plus;
   Parsing::Rule r_Expr_preinc;
   Parsing::Rule r_Expr_postinc;
   Parsing::Rule r_Expr_predec;
   Parsing::Rule r_Expr_postdec;
   Parsing::Rule r_Expr_minus;
   Parsing::Rule r_Expr_pars;
   Parsing::Rule r_Expr_pars2;
   Parsing::Rule r_Expr_times;
   Parsing::Rule r_Expr_div;
   Parsing::Rule r_Expr_mod;
   Parsing::Rule r_Expr_pow;
   Parsing::Rule r_Expr_shl;
   Parsing::Rule r_Expr_shr;
   
   Parsing::Rule r_Expr_and;
   Parsing::Rule r_Expr_or;
   
   Parsing::Rule r_Expr_band;
   Parsing::Rule r_Expr_bor;
   Parsing::Rule r_Expr_bxor;
   Parsing::Rule r_Expr_bnot;
   
   Parsing::Rule r_Expr_oob;
   Parsing::Rule r_Expr_deoob;
   Parsing::Rule r_Expr_xoob;
   Parsing::Rule r_Expr_isoob;
   Parsing::Rule r_Expr_str_ipol;

   Parsing::Rule r_Expr_expr_evalret;
   Parsing::Rule r_Expr_expr_evalret_exec;
   Parsing::Rule r_Expr_expr_evalret_doubt;
   Parsing::Rule r_Expr_provides;
   
   Parsing::Rule r_Expr_named;

   Parsing::Rule r_Expr_auto_add;
   Parsing::Rule r_Expr_auto_sub;
   Parsing::Rule r_Expr_auto_times;
   Parsing::Rule r_Expr_auto_div;
   Parsing::Rule r_Expr_auto_mod;
   Parsing::Rule r_Expr_auto_pow;
   Parsing::Rule r_Expr_auto_shr;
   Parsing::Rule r_Expr_auto_shl;
   Parsing::Rule r_Expr_invoke;
   Parsing::Rule r_Expr_expr_compose;
   Parsing::Rule r_Expr_expr_funcpower;
   
   Parsing::Rule r_Expr_ternary_if;
   Parsing::Rule r_Expr_expr_eval;
   Parsing::Rule r_Expr_expr_lit;
   Parsing::Rule r_Expr_expr_unquote;
   
   Parsing::Rule r_Expr_neg;
   Parsing::Rule r_Expr_neg2;
   Parsing::Rule r_Expr_not;
   
   Parsing::Rule r_Expr_Atom;

   Parsing::Rule r_Expr_function;
   Parsing::Rule r_Expr_functionEta;
   Parsing::Rule r_Expr_lambda;
   Parsing::Rule r_Expr_ep;
   Parsing::Rule r_Expr_accumulator;
   Parsing::Rule r_Expr_class;
   Parsing::Rule r_Expr_proto;
   Parsing::Rule r_Expr_lit;
   Parsing::Rule r_Expr_parametric_lit;

   //================================================
   // Function
   //

   Parsing::NonTerminal S_Function;
   Parsing::Rule r_function_short;
   Parsing::Rule r_function;
   Parsing::Rule r_function_eta;

   Parsing::NonTerminal S_Return;
   Parsing::Rule r_return_doubt;
   Parsing::Rule r_return_eval;
   Parsing::Rule r_return_break;
   Parsing::Rule r_return_expr;
   Parsing::Rule r_return;

   Parsing::NonTerminal S_Class;
   Parsing::Rule r_class;

   Parsing::NonTerminal S_Object;
   Parsing::Rule r_object;

   Parsing::Rule r_class_from;
   Parsing::Rule r_class_pure;
   Parsing::Rule r_class_p_from;
   Parsing::Rule r_class_p;
   
   Parsing::NonTerminal S_InitDecl;
   Parsing::Rule r_init;

   Parsing::NonTerminal FromClause;
   Parsing::Rule r_FromClause_next;
   Parsing::Rule r_FromClause_first;
   Parsing::Rule r_FromClause_empty;

   Parsing::NonTerminal FromEntry;
   Parsing::Rule r_FromClause_entry_with_expr;
   Parsing::Rule r_FromClause_entry;

   Parsing::NonTerminal S_PropDecl;
   Parsing::Rule r_propdecl_expr;
   Parsing::Rule r_propdecl_simple;

   Parsing::NonTerminal S_StaticPropDecl;
   Parsing::Rule r_propdecl_static_expr;
   Parsing::Rule r_static_function;
   Parsing::Rule r_static_function_eta;


   //================================================
   // Atom
   //
   Parsing::NonTerminal Atom;
   Parsing::Rule r_Atom_Int;
   Parsing::Rule r_Atom_Float;
   Parsing::Rule r_Atom_Name;
   Parsing::Rule r_Atom_Pure_Name;
   Parsing::Rule r_Atom_String;
   Parsing::Rule r_Atom_RString;
   Parsing::Rule r_Atom_IString;
   Parsing::Rule r_Atom_MString;
   Parsing::Rule r_Atom_False;
   Parsing::Rule r_Atom_True;
   Parsing::Rule r_Atom_self;
   Parsing::Rule r_Atom_fself;
   Parsing::Rule r_Atom_init;
   Parsing::Rule r_Atom_continue;
   Parsing::Rule r_Atom_break;
   Parsing::Rule r_Atom_Nil;

   //================================================
   // Expression lists
   //
   Parsing::NonTerminal ListExpr;
   Parsing::Rule r_ListExpr_next;
   Parsing::Rule r_ListExpr_next_no_comma;
   Parsing::Rule r_ListExpr_nextd;
   Parsing::Rule r_ListExpr_first;
   Parsing::Rule r_ListExpr_eol;
   Parsing::Rule r_ListExpr_empty;
   
   Parsing::NonTerminal CaseListRange;
   Parsing::Rule r_CaseListRange_int;
   Parsing::Rule r_CaseListRange_string;
      
   Parsing::NonTerminal CaseListToken;
   Parsing::Rule r_CaseListToken_range;
   Parsing::Rule r_CaseListToken_nil;
   Parsing::Rule r_CaseListToken_true;
   Parsing::Rule r_CaseListToken_false;
   Parsing::Rule r_CaseListToken_int;
   Parsing::Rule r_CaseListToken_string;
   Parsing::Rule r_CaseListToken_rstring;
   Parsing::Rule r_CaseListToken_sym;
   
   
   Parsing::NonTerminal CaseList;
   Parsing::Rule r_CaseList_next;
   Parsing::Rule r_CaseList_first;

   Parsing::NonTerminal NeListExpr;
   Parsing::Rule r_NeListExpr_next;
   Parsing::Rule r_NeListExpr_first;

   Parsing::NonTerminal NeListExpr_ungreed;
   Parsing::Rule r_NeListExpr_ungreed_next;
   Parsing::Rule r_NeListExpr_ungreed_first;

   //================================================
   // Symbol list
   //

   Parsing::NonTerminal ListSymbol;
   Parsing::Rule r_ListSymbol_eol;
   Parsing::Rule r_ListSymbol_nextd;
   Parsing::Rule r_ListSymbol_next;
   Parsing::Rule r_ListSymbol_first;
   Parsing::Rule r_ListSymbol_empty;

   Parsing::NonTerminal NeListSymbol;
   Parsing::Rule r_NeListSymbol_next;
   Parsing::Rule r_NeListSymbol_first;

   Parsing::NonTerminal LambdaParams;
   Parsing::Rule r_lit_params;
   Parsing::Rule r_lit_params_eta;
   Parsing::Rule r_lambda_params;
   Parsing::Rule r_lambda_params_eta;

   Parsing::NonTerminal EPBody;
   Parsing::Rule r_lit_epbody;

   Parsing::NonTerminal AccumulatorBody;
   Parsing::Rule r_accumulator_complete;
   Parsing::Rule r_accumulator_w_filter;
   Parsing::Rule r_accumulator_w_target;
   Parsing::Rule r_accumulator_simple;

   Parsing::NonTerminal ClassParams;
   Parsing::NonTerminal ObjectParams;
   
   Parsing::NonTerminal S_ProtoProp;
   Parsing::Rule r_proto_prop;

   //================================================
   // Arrays and dictionaries.
   //

   Parsing::NonTerminal ArrayEntry;
   Parsing::Rule r_array_entry_expr2;
   Parsing::Rule r_array_entry_expr1;
   Parsing::Rule r_array_entry_comma;
   Parsing::Rule r_array_entry_eol;
   Parsing::Rule r_array_entry_arrow;
   Parsing::Rule r_array_entry_close;
   Parsing::Rule r_array_entry_range3;
   Parsing::Rule r_array_entry_range3bis;
   Parsing::Rule r_array_entry_range2;
   Parsing::Rule r_array_entry_range1;
   Parsing::Rule r_array_entry_runaway;

   Parsing::NonTerminal UnboundKeyword;
   Parsing::Rule r_uk_if;
   Parsing::Rule r_uk_elif;
   Parsing::Rule r_uk_else;
   Parsing::Rule r_uk_while;
   Parsing::Rule r_uk_for;
   
   //================================================
   // States
   //

   Parsing::State s_Main;
   Parsing::State s_InlineFunc;
   Parsing::State s_ClassBody;
   Parsing::State s_LambdaStart;
   Parsing::State s_EPState;
   Parsing::State s_ClassStart;
   Parsing::State s_ObjectStart;
   Parsing::State s_ProtoDecl;
   Parsing::State s_ArrayDecl;
};

}

#endif	/* SOURCEPARSER_H */

/* end of sourceparser.h */
