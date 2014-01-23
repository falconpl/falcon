/*
   FALCON - The Falcon Programming Language.
   FILE: sourceparser.cpp

   Parser for Falcon source files.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Apr 2011 00:04:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/sourceparser.cpp"

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/stderrors.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>

#include <falcon/sp/parser_arraydecl.h>
#include <falcon/sp/parser_attribute.h>
#include <falcon/sp/parser_assign.h>
#include <falcon/sp/parser_atom.h>
#include <falcon/sp/parser_autoexpr.h>
#include <falcon/sp/parser_accumulator.h>
#include <falcon/sp/parser_class.h>
#include <falcon/sp/parser_call.h>
#include <falcon/sp/parser_dynsym.h>
#include <falcon/sp/parser_end.h>
#include <falcon/sp/parser_export.h>
#include <falcon/sp/parser_expr.h>
#include <falcon/sp/parser_fastprint.h>
#include <falcon/sp/parser_for.h>
#include <falcon/sp/parser_function.h>
#include <falcon/sp/parser_if.h>
#include <falcon/sp/parser_index.h>
#include <falcon/sp/parser_import.h>
#include <falcon/sp/parser_list.h>
#include <falcon/sp/parser_load.h>
#include <falcon/sp/parser_namespace.h>
#include <falcon/sp/parser_global.h>
#include <falcon/sp/parser_proto.h>
#include <falcon/sp/parser_rule.h>
#include <falcon/sp/parser_switch.h>
#include <falcon/sp/parser_summon.h>
#include <falcon/sp/parser_ternaryif.h>
#include <falcon/sp/parser_try.h>
#include <falcon/sp/parser_while.h>
#include <falcon/sp/parser_loop.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/lexer.h>
#include <falcon/parser/parser.h>

#include <falcon/error.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

static void apply_dummy( const Rule&, Parser& p )
{
   p.simplify(1);
}

//==========================================================
// SourceParser
//==========================================================

SourceParser::SourceParser():
   T_Openpar("(",20),
   T_Closepar(")"),
   T_OpenSquare("[", 20),
   T_CapPar("^("),
   T_CapSquare("^["),
   T_DotPar(".("),
   T_DotSquare(".["),
   T_CloseSquare("]"),
   T_OpenGraph("{",20),
   T_OpenProto("p{"),
   T_CloseGraph("}"),

   T_Dot(".",15),
   T_Arrow("=>", 170 ),
   T_AutoAdd( "+=", 70 ),
   T_AutoSub( "-=", 70 ),
   T_AutoTimes( "*=", 70 ),
   T_AutoDiv( "/=", 70 ),
   T_AutoMod( "%=", 70 ),
   T_AutoPow( "**=", 70 ),
   T_AutoRShift( ">>=", 70 ),
   T_AutoLShift( "<<=", 70 ),
   T_EEQ( "===", 70 ),
   
   T_BAND("^&", 60),
   T_BOR("^|", 65),
   T_BXOR("^^", 65),
   T_BNOT("^!", 23),
   
   T_OOB("^+", 24),
   T_DEOOB("^-", 24),
   T_XOOB("^%", 24),
   T_ISOOB("^$", 24),
   T_UNQUOTE("^~", 10 ),
   T_COMPOSE("^.", 60),
   T_EVALRET( "^=", 150),
   T_EVALRET_EXEC( "^*", 150),
   T_EVALRET_DOUBT( "^?", 150),
   T_STARARROW( "*=>", 170),

   T_Comma( "," , 180 ),
   T_QMark( "?" , 175, true ),
   T_Tilde( "~" , 5 ),
   T_Bang("!"),
   T_Disjunct( "|" , 130 ),

   T_UnaryMinus("(neg)",23),
   T_Dollar("$",23),
   T_Amper("&",10),
   T_NumberSign("#", 68),   /* between bit and & equality */
   T_At("@", 10),           /* Must have a very high priority,
                               as @"abc".x should be interpreted as (@"abc").x */

   T_Power("**", 25),

   T_Times("*",30),
   T_Divide("/",30),
   T_Modulo("%",30),
   T_RShift(">>",30),
   T_LShift("<<",30),

   T_Plus("+",50),
   T_Minus("-",50),
   T_PlusPlus("++",21, true),
   T_MinusMinus("--",21, true),

   T_DblEq("==", 70),
   T_NotEq("!=", 70),
   T_Less("<", 70),
   T_Greater(">", 70),
   T_LE("<=", 70),
   T_GE(">=", 70),
   T_Colon( ":", 170 ),
   T_EqSign("=", 200, false),
   T_EqSign2("=", 200 ),


   T_as("as"),
   T_if("if"),
   T_in("in", 23),
   T_or("or", 130),
   T_to("to", 70),

   T_and("and", 120),
   T_def("def"),
   T_end("end"),
   T_for("for"),
   T_not("not", 50),
   T_nil("nil"),
   T_try("try"),
   T_catch("catch"),
   T_notin("notin", 23),
   T_finally("finally"),
   T_raise("raise"),
   T_fself("fself"),

   T_elif("elif"),
   T_else("else"),
   T_rule("rule"),

   T_while("while"),

   T_function("function"),
   T_return("return"),
   T_class("class"),
   T_object("object"),
   T_init("init"),

   T_true( "true" ),
   T_false( "false" ),
   T_self( "self" ),
   T_from( "from" ),
   T_load( "load" ),
   T_export( "export" ),
   T_import( "import" ),
   T_namespace( "namespace" ),
   T_global("global"),
   
   T_forfirst( "forfirst" ),
   T_formiddle( "formiddle" ),
   T_forlast( "forlast" ),
   T_break( "break" ),
   T_continue( "continue" ),
   
   T_switch("switch"),
   T_case("case"),
   T_default("default"),
   T_select("select"),
   T_loop("loop"),

   T_RString("R-String"),
   T_IString("I-String"),
   T_MString("M-String"),
   T_provides("provides"),
   
   T_DoubleColon("::",16),
   T_ColonQMark(":?",16)
{
   S_Attribute << "Attribute" << errhand_attribute;
   S_Attribute << (r_attribute << "Attribute" << apply_attribute << T_Colon << T_Name <<  T_Arrow << Expr << T_EOL);

   S_Autoexpr << "Autoexpr"
      << (r_line_autoexpr << "Autoexpr" << apply_line_expr << Expr << T_EOL)
      << (r_assign_list << "Autoexpr_list" << apply_autoexpr_list << S_MultiAssign )
      ;

   S_If << "IF" << errhand_if;
   S_If << (r_if_short << "if_short" << apply_if_short << T_if << Expr << T_Colon );
   S_If << (r_if << "if" << apply_if << T_if << Expr << T_EOL );

   S_Elif << "ELIF"
      << (r_elif << "elif" << apply_elif << T_elif << Expr << T_EOL )
      ;

   S_Else << "ELSE"
      << (r_else << "else" << apply_else << T_else << T_EOL )
      ;

   S_While << "WHILE" << while_errhand
      << (r_while_short << "while_short" << apply_while_short << T_while << Expr << T_Colon )
      << (r_while << "while" << apply_while << T_while << Expr << T_EOL )
      ;
   
   S_Loop << "LOOP" << loop_errhand;
   S_Loop << ( r_loop_short << "loop short" << apply_loop_short << T_loop << T_Colon );
   S_Loop << ( r_loop << "loop" << apply_loop << T_loop << T_EOL );

   S_For << "FOR" << for_errhand;
   S_For << (r_for_to_step << "FOR/to/step" << apply_for_to_step
            << T_for << T_Name << T_EqSign << Expr << T_to << Expr << T_Comma << Expr << T_EOL )
      << (r_for_to_step_short << "FOR/to/step short" << apply_for_to_step_short
            << T_for << T_Name << T_EqSign << Expr << T_to << Expr << T_Comma << Expr << T_Colon )
      << (r_for_to << "FOR/to" << apply_for_to 
            << T_for << T_Name << T_EqSign << Expr << T_to << Expr << T_EOL )
      << (r_for_to_short << "FOR/to short" << apply_for_to_short
            << T_for << T_Name << T_EqSign << Expr << T_to << Expr << T_Colon )
      << (r_for_in << "FOR/in" << apply_for_in << T_for << NeListSymbol << T_in << Expr << T_EOL )
      << (r_for_in_short << "FOR/in short" << apply_for_in_short << T_for << NeListSymbol << T_in << Expr << T_Colon )
      ;
      
   S_Forfirst << "forfirst";
   S_Forfirst << (r_forfirst << "FORfirst" << apply_forfirst
            << T_forfirst << T_EOL )     
      << (r_forfirst_short << "FORfirst short" << apply_forfirst_short
            << T_forfirst << T_Colon )     
      ;
   
   S_Formiddle << "formiddle";
   S_Formiddle << (r_formiddle << "FORmiddle" << apply_formiddle
            << T_formiddle << T_EOL )     
         << (r_formiddle_short << "FORmiddle short" << apply_formiddle_short
            << T_formiddle << T_Colon )     
      ;

   S_Forlast << "forlast";
   S_Forlast << (r_forlast << "FORlast" << apply_forlast
            << T_forlast << T_EOL )
       << (r_forlast_short << "FORlast short" << apply_forlast_short
            << T_forlast << T_Colon )
      ;
   
   S_Switch << "switch";
   S_Switch << (r_switch << "r_switch" << apply_switch << T_switch << Expr << T_EOL )
      ;
   
   S_Select << "select";
   S_Select << (r_select << "r_select" << apply_select << T_select << Expr << T_EOL )
      ;
   
   S_Case << "case";
   S_Case << (r_case << "r_case" << apply_case << T_case << CaseList << T_EOL )
          << (r_case_short << "r_case_short" << apply_case_short << T_case << CaseList << T_Colon )
      ;
   
   S_Default << "default";
   S_Default << (r_default << "r_default" << apply_default << T_default << T_EOL )
             << (r_default_short << "r_default_short" << apply_default_short << T_default << T_Colon )
      ;

   S_Cut << "CUT"
      << (r_cut_expr << "cut-expr" << apply_cut_expr << T_Bang << Expr << T_EOL )
      << (r_cut << "cut" << apply_cut << T_Bang << T_EOL )
      ;

   S_Doubt << "Doubt"
      << (r_doubt << "doubt-expr" << apply_doubt << T_QMark << Expr << T_EOL )
      ;
   
   S_End << "END"
      << (r_end_rich << "RichEnd" << apply_end_rich << T_end << Expr << T_EOL )
      << (r_end << "end" << apply_end << T_end << T_EOL)
      ;

   S_SmallEnd << "Small END"
      << (r_end_small << "end_small" << apply_end_small << T_end )
      ;

   S_EmptyLine << "EMPTY"
      << (r_empty << "Empty line" << apply_dummy << T_EOL )
      ;
   
   S_Continue << "Continue"
      << (r_continue << "continue rule" << apply_continue << T_continue << T_EOL )
      ;
   
   S_Break << "Break"
      << (r_break << "break rule" << apply_break << T_break << T_EOL )
      ;

   S_MultiAssign << "MultiAssign"
      //<< (r_Stmt_assign_list << "STMT_assign_list" << apply_stmt_assign_list << NeListExpr_ungreed << T_EqSign << NeListExpr )
            << (r_Stmt_assign_list << "STMT_assign_list" << apply_stmt_assign_list << NeListExpr << T_EqSign << NeListExpr << T_EOL )
      ;

   S_FastPrint << "FastPrint";
   S_FastPrint << PrintExpr_errhand;
   S_FastPrint << ( r_fastprint_alone << "fastprint alone rule" << apply_fastprint_alone << T_RShift << T_EOL );
   S_FastPrint << ( r_fastprint << "fastprint rule" << apply_fastprint << T_RShift << ListExpr << T_EOL );
   S_FastPrint << ( r_fastprint_nl_alone << "fastprint+nl alone" << apply_fastprint_nl_alone << T_Greater << T_EOL );
   S_FastPrint << ( r_fastprint_nl << "fastprint+nl" << apply_fastprint_nl << T_Greater << ListExpr << T_EOL );

   S_Load << "load" << load_errhand;
   S_Load << (r_load_string << "load_string" << apply_load_string << T_load << T_String << T_EOL );
   S_Load << (r_load_mod_spec << "load_mod_spec" << apply_load_mod_spec << T_load << ModSpec << T_EOL );

   ModSpec << "ModSpec" << load_modspec_errhand;
   ModSpec << (r_modspec_next << "modspec_next" << apply_modspec_next << ModSpec << T_Dot << T_Name );
   ModSpec << (r_modspec_first << "modspec_first" << apply_modspec_first << T_Name );
   ModSpec << (r_modspec_first_self << "modspec_first_self" << apply_modspec_first_self << T_self );
   ModSpec << (r_modspec_first_dot << "modspec_first_dot" << apply_modspec_first_dot << T_Dot );

   S_Export << "export" << export_errhand;
   S_Export << ( r_export_rule << "export_rule" << apply_export_rule << T_export << ListSymbol << T_EOL );
   S_Global << ( r_global_rule << "global_rule" << apply_global << T_global << ListSymbol << T_EOL );
   
   S_Try << "try" << try_errhand;
   S_Try << ( r_try_rule << "try_rule" << apply_try << T_try << T_EOL );
   S_Catch << "catch" << catch_errhand;
   S_Catch << ( r_catch_base << "catch_" << apply_catch << T_catch << CatchSpec ); 
   S_Finally << "finally" << finally_errhand;
   S_Finally << ( r_finally << "r_finally" << apply_finally << T_finally << T_EOL ); 
   
   CatchSpec << "CatchSpec";
   CatchSpec << ( r_catch_all << "catch_all" << apply_catch_all << T_EOL );
   CatchSpec << ( r_catch_in_var<< "catch_in_var" << apply_catch_in_var << T_in << T_Name << T_EOL );
   CatchSpec << ( r_catch_as_var<< "catch_as_var" << apply_catch_as_var << T_as << T_Name << T_EOL );
   CatchSpec << ( r_catch_thing << "catch_thing" << apply_catch_thing <<  CaseList << T_EOL );
   CatchSpec << ( r_catch_thing_in_var << "catch_thing_in_var" << apply_catch_thing_in_var << CaseList << T_in << T_Name << T_EOL );
   CatchSpec << ( r_catch_thing_as_var << "catch_thing_as_var" << apply_catch_thing_as_var << CaseList << T_as << T_Name << T_EOL );
   
   S_Raise << "raise" << raise_errhand;
   S_Raise << ( r_raise << "r_raise" << apply_raise << T_raise << Expr << T_EOL );
   
   //==========================================================================
   // Import & family
   //
   
   // We use a selector strategy to reduce the amount of tokens visible to the root state.s
   S_Import << "import" << import_errhand;
   S_Import << ( r_import_rule << "import_rule" << apply_import << T_import << ImportClause << T_EOL );
   
   ImportClause << "ImportClause";
   ImportClause << ( r_import_star_from_string_in << "import_star_from_string_in" << apply_import_star_from_string_in
      << T_Times << T_from << T_String << T_in << NameSpaceSpec << T_EOL );
   ImportClause << ( r_import_star_from_string << "import_star_from_string" << apply_import_star_from_string
      << T_Times << T_from << T_String << T_EOL );
   
   ImportClause << ( r_import_star_from_modspec_in << "import_star_from_modspec_in" << apply_import_star_from_modspec_in
      << T_Times << T_from << ModSpec << T_in << NameSpaceSpec << T_EOL );
   ImportClause << ( r_import_star_from_modspec << "import_star_from_modspec_in" << apply_import_star_from_modspec
      << T_Times << T_from << ModSpec << T_EOL );
   
   ImportClause << ( r_import_from_string_as << "import_from_string_as" << apply_import_from_string_as
      << ImportSpec << T_from << T_String << T_as << T_Name << T_EOL );
   ImportClause << ( r_import_from_string_in << "import_from_string_in" << apply_import_from_string_in
      << ImportSpec << T_from << T_String << T_in << NameSpaceSpec << T_EOL );
   ImportClause << ( r_import_from_string << "import_from_string" << apply_import_string
      << ImportSpec << T_from << T_String << T_EOL );
   
   ImportClause << ( r_import_from_modspec_as << "import_from_modspec_as" << apply_import_from_modspec_as
      << ImportSpec << T_from << ModSpec << T_as << T_Name << T_EOL );
   ImportClause << ( r_import_from_modspec_in << "import_from_modspec_in" << apply_import_from_modspec_in
      << ImportSpec << T_from << ModSpec << T_in << NameSpaceSpec << T_EOL );
   ImportClause << ( r_import_from_modspec << "import_from_modspec" << apply_import_from_modspec
      << ImportSpec << T_from << ModSpec << T_EOL );

   ImportClause << ( r_import_syms << "import_syms" << apply_import_syms
      << ImportSpec << T_EOL );


   ImportSpec << "ImportSpec" << importspec_errhand;
   ImportSpec<< (r_ImportSpec_next <<  "ImportSpec_next" <<  apply_ImportSpec_next << ImportSpec << T_Comma << NameSpaceSpec );
   ImportSpec<< (r_ImportSpec_attach_last <<  "ImportSpec_attach_last" <<  apply_ImportSpec_attach_last << ImportSpec << T_Dot << T_Times );
   ImportSpec<< (r_ImportSpec_attach_next <<  "ImportSpec_attach_next" <<  apply_ImportSpec_attach_next << ImportSpec << T_Dot << T_Name );
   ImportSpec<< (r_ImportSpec_first << "ImportSpec_first" << apply_ImportSpec_first << NameSpaceSpec );
   ImportSpec<< (r_ImportSpec_empty << "ImportSpec_empty" << apply_ImportSpec_empty );
   
   NameSpaceSpec << "NameSpaceSpec";
   NameSpaceSpec << (r_NameSpaceSpec_last <<  "NameSpaceSpec_last" <<  apply_nsspec_last << NameSpaceSpec << T_Dot << T_Times );
   NameSpaceSpec << (r_NameSpaceSpec_next <<  "NameSpaceSpec_next" <<  apply_nsspec_next << NameSpaceSpec << T_Dot << T_Name );
   NameSpaceSpec << (r_NameSpaceSpec_first << "NameSpaceSpec_first" << apply_nsspec_first << T_Name );

   S_Namespace << "Namespace decl" << namespace_errhand;
   S_Namespace << (r_NameSpace <<  "NameSpace" <<  apply_namespace << T_namespace << NameSpaceSpec << T_EOL );
   
   //==========================================================================
   // Expression
   //
   Expr << "Expr";
   Expr << expr_errhand;

   // Unary operators
   // the lexer may find a non-unary minus when parsing it not after an operator...;   
   Expr<< (r_Expr_neg   << "Expr_neg"   << apply_expr_neg << T_Minus << Expr );
   // ... or find an unary minus when getting it after another operator.;
   Expr<< (r_Expr_neg2   << "Expr_neg2"   << apply_expr_neg << T_UnaryMinus << Expr );   
   Expr<< (r_Expr_not   << "Expr_not"  << apply_expr_not  << T_not << Expr );
   Expr<< (r_Expr_bnot  << "Expr_Bnot" << apply_expr_bnot << T_BNOT << Expr );

   Expr<< (r_Expr_oob  << "Expr_oob" << apply_expr_oob << T_OOB << Expr );
   Expr<< (r_Expr_deoob  << "Expr_deoob" << apply_expr_deoob << T_DEOOB << Expr );
   Expr<< (r_Expr_xoob  << "Expr_xoob" << apply_expr_xoob << T_XOOB << Expr );
   Expr<< (r_Expr_isoob  << "Expr_isoob" << apply_expr_isoob << T_ISOOB << Expr );
   Expr<< (r_Expr_str_ipol  << "Expr_str_ipol" << apply_expr_str_ipol << T_At << Expr );

   Expr<< (r_Expr_expr_evalret << "Expr_evarlet"  << apply_expr_evalret << T_EVALRET << Expr );
   Expr<< (r_Expr_expr_evalret_exec << "Expr_evarlet_exec"  << apply_expr_evalret_exec << T_EVALRET_EXEC << Expr );
   Expr<< (r_Expr_expr_evalret_doubt << "Expr_evarlet_doubt"  << apply_expr_evalret_doubt << T_EVALRET_DOUBT << Expr );

   Expr<< (r_Expr_named << "Expr named" << apply_expr_named << T_Name << T_Disjunct << Expr);
   Expr<< (r_Expr_provides << "Expr_provides" << apply_expr_provides << Expr << T_provides << T_Name);

   Expr << (r_Expr_assign << "Expr_assign" << apply_expr_assign << Expr << T_EqSign << Expr );


   Expr<< (r_Expr_equal << "Expr_equal" << apply_expr_equal << Expr << T_DblEq << Expr);
   Expr<< (r_Expr_diff << "Expr_diff" << apply_expr_diff << Expr << T_NotEq << Expr);
   Expr<< (r_Expr_less << "Expr_less" << apply_expr_less << Expr << T_Less << Expr);
   Expr<< (r_Expr_greater << "Expr_greater" << apply_expr_greater << Expr << T_Greater << Expr);
   Expr<< (r_Expr_le << "Expr_le" << apply_expr_le << Expr << T_LE << Expr);
   Expr<< (r_Expr_ge << "Expr_ge" << apply_expr_ge << Expr << T_GE << Expr);
   Expr<< (r_Expr_eeq << "Expr_eeq" << apply_expr_eeq << Expr << T_EEQ << Expr);
   Expr<< (r_Expr_in << "Expr in" << apply_expr_in << Expr << T_in << Expr);
   Expr<< (r_Expr_notin << "Expr notin" << apply_expr_notin << Expr << T_notin << Expr);

   Expr<< (r_Expr_call << "Expr_call" << apply_expr_call << Expr << T_Openpar << ListExpr << T_Closepar );
   Expr<< (r_Expr_summon << "Expr_summon" << apply_expr_summon << Expr << T_DoubleColon << T_Name << T_OpenSquare << ListExpr << T_CloseSquare );
   Expr<< (r_Expr_opt_summon << "Expr_opt_summon" << apply_expr_opt_summon << Expr << T_ColonQMark << T_Name << T_OpenSquare <<  ListExpr << T_CloseSquare );

   Expr<< (r_Expr_index << "Expr_index" << apply_expr_index << Expr << T_OpenSquare << Expr << T_CloseSquare );
   Expr<< (r_Expr_star_index << "Expr_star_index" << apply_expr_star_index << Expr << T_OpenSquare << T_Times << Expr << T_CloseSquare );
   Expr<< (r_Expr_range_index3 << "Expr_Expr_range_index3" << apply_expr_range_index3
            << Expr << T_OpenSquare << Expr << T_Colon << Expr << T_Colon << Expr << T_CloseSquare );
   Expr<< (r_Expr_range_index3open << "Expr_Expr_range_index3open" << apply_expr_range_index3open
            << Expr << T_OpenSquare << Expr << T_Colon << T_Colon << Expr << T_CloseSquare );
   Expr<< (r_Expr_range_index2 << "Expr_Expr_range_index2" << apply_expr_range_index2
            << Expr << T_OpenSquare << Expr << T_Colon << Expr << T_CloseSquare );
   Expr<< (r_Expr_range_index1 << "Expr_Expr_range_index1" << apply_expr_range_index1
            << Expr << T_OpenSquare << Expr << T_Colon << T_CloseSquare );
   Expr<< (r_Expr_range_index0 << "Expr_Expr_range_index0" << apply_expr_range_index0
            << Expr << T_OpenSquare << T_Colon << T_CloseSquare );
   
   Expr<< (r_Expr_array_decl << "Expr_array_decl" << apply_expr_array_decl << T_OpenSquare );
   Expr<< (r_Expr_array_decl2 << "Expr_array_decl2" << apply_expr_array_decl2 << T_DotSquare );
   Expr<< (r_Expr_accumulator << "Expr_accumulator" << apply_expr_accumulator << T_CapSquare << AccumulatorBody );
   
   Expr<< (r_Expr_amper << "Expr_dyns" << apply_expr_amper << T_Amper << T_Name );
   
   Expr<< (r_Expr_dot << "Expr_dot" << apply_expr_dot << Expr << T_Dot << T_Name);
   Expr<< (r_Expr_plus << "Expr_plus" << apply_expr_plus << Expr << T_Plus << Expr);
   Expr<< (r_Expr_preinc << "Expr_preinc" << apply_expr_preinc << T_PlusPlus << Expr);
   Expr<< (r_Expr_postinc << "Expr_postinc" << apply_expr_postinc << Expr << T_PlusPlus);
   Expr<< (r_Expr_predec << "Expr_predec" << apply_expr_predec << T_MinusMinus << Expr);
   Expr<< (r_Expr_postdec << "Expr_postdec" << apply_expr_postdec << Expr << T_MinusMinus);
   
   Expr<< (r_Expr_minus << "Expr_minus" << apply_expr_minus << Expr << T_Minus << Expr);
   Expr<< (r_Expr_pars << "Expr_pars" << apply_expr_pars << T_Openpar << Expr << T_Closepar);
   Expr<< (r_Expr_pars2 << "Expr_pars2" << apply_expr_pars << T_DotPar << Expr << T_Closepar);
   Expr<< (r_Expr_times << "Expr_times" << apply_expr_times << Expr << T_Times << Expr);
   Expr<< (r_Expr_div   << "Expr_div"   << apply_expr_div   << Expr << T_Divide << Expr );
   Expr<< (r_Expr_mod   << "Expr_mod"   << apply_expr_mod   << Expr << T_Modulo << Expr );
   Expr<< (r_Expr_pow   << "Expr_pow"   << apply_expr_pow   << Expr << T_Power << Expr );
   Expr<< (r_Expr_shr   << "Expr_shr"   << apply_expr_shr   << Expr << T_RShift << Expr );
   Expr<< (r_Expr_shl   << "Expr_shl"   << apply_expr_shl   << Expr << T_LShift << Expr );

   Expr<< (r_Expr_and << "Expr_and" << apply_expr_and  << Expr << T_and << Expr );
   Expr<< (r_Expr_or  << "Expr_or"  << apply_expr_or   << Expr << T_or << Expr );
   
   Expr<< (r_Expr_band << "Expr_band" << apply_expr_band  << Expr << T_BAND << Expr );
   Expr<< (r_Expr_bor  << "Expr_bor"  << apply_expr_bor   << Expr << T_BOR << Expr );
   Expr<< (r_Expr_bxor << "Expr_bxor" << apply_expr_bxor  << Expr << T_BXOR << Expr );
   
   Expr<< (r_Expr_auto_add << "Expr_auto_add"   << apply_expr_auto_add   << Expr << T_AutoAdd << Expr );
   Expr<< (r_Expr_auto_sub << "Expr_auto_sub"   << apply_expr_auto_sub   << Expr << T_AutoSub << Expr );
   Expr<< (r_Expr_auto_times << "Expr_auto_times"   << apply_expr_auto_times   << Expr << T_AutoTimes << Expr );
   Expr<< (r_Expr_auto_div << "Expr_auto_div"   << apply_expr_auto_div   << Expr << T_AutoDiv << Expr );
   Expr<< (r_Expr_auto_mod << "Expr_auto_mod"   << apply_expr_auto_mod   << Expr << T_AutoMod << Expr );
   Expr<< (r_Expr_auto_pow << "Expr_auto_pow"   << apply_expr_auto_pow   << Expr << T_AutoPow << Expr );
   Expr<< (r_Expr_auto_shl << "Expr_auto_shr"   << apply_expr_auto_shr  << Expr << T_AutoRShift << Expr );
   Expr<< (r_Expr_auto_shr << "Expr_auto_shl"   << apply_expr_auto_shl   << Expr << T_AutoLShift << Expr );   

   Expr<< (r_Expr_invoke << "Expr_invoke"   << apply_expr_invoke   << Expr << T_NumberSign << Expr );
   Expr<< (r_Expr_expr_compose << "Expr_compose"  << apply_expr_compose << Expr << T_COMPOSE << Expr );
   
   Expr<< (r_Expr_ternary_if << "Expr_ternary_if"   << apply_expr_ternary_if  
            << Expr << T_QMark << Expr << T_Colon << Expr );
   r_Expr_ternary_if.setGreedy(true);
   
   Expr<< (r_Expr_expr_unquote << "Expr_unquote"  << apply_expr_unquote << T_UNQUOTE << Expr );
   
   
   Expr<< (r_Expr_Atom << "Expr_atom" << apply_expr_atom << Atom);
   Expr<< (r_Expr_function << "Expr_func" << apply_expr_func << T_function << T_Openpar << ListSymbol << T_Closepar << T_EOL);
   Expr<< (r_Expr_functionEta << "Expr_funcEta" << apply_expr_funcEta << T_function << T_Times << T_Openpar << ListSymbol << T_Closepar << T_EOL);
   // Start of lambda expressions.
   Expr<< (r_Expr_lambda << "Expr_lambda" << apply_expr_lambda << T_OpenGraph );
   Expr<< (r_Expr_ep << "Expr_ep" << apply_expr_ep << T_CapPar );
   Expr<< (r_Expr_class << "Expr_class" << apply_expr_class << T_class );
   Expr<< (r_Expr_proto << "Expr_proto" << apply_expr_proto << T_OpenProto );
   Expr<< (r_rule << "Rule_rule" << apply_rule << T_rule << T_EOL );


   S_RuleOr << (r_rule_or << "Rule_rule_or" << apply_rule_branch << T_or << T_EOL );

   S_Function << "Function"
      /* This requires a bit of work << (r_function_short << "Function short" << apply_function_short
            << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar <<  T_Colon << Expr << T_EOL )
       */
      << (r_function << "Function decl" << apply_function
             << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL )
      << (r_function_eta << "Function ETA decl" << apply_function_eta
             << T_function << T_Times << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL )
      ;
      
   S_Return << "Return"
      << (r_return_doubt << "return doubt" << apply_return_doubt << T_return << T_QMark << Expr << T_EOL)
      << (r_return_eval << "return eval" << apply_return_eval << T_return << T_Times << Expr << T_EOL)
      << (r_return_break << "return break" << apply_return_break << T_return << T_break << T_EOL)
      << (r_return_expr << "return expr" << apply_return_expr << T_return << Expr << T_EOL)
      << (r_return << "return" << apply_return << T_return << T_EOL)
      ;

   Atom << "Atom"
      << (r_Atom_Int << "Atom_Int" << apply_Atom_Int << T_Int )
      << (r_Atom_Float << "Atom_Float" << apply_Atom_Float << T_Float )
      << (r_Atom_Name << "Atom_Name" << apply_Atom_Name << T_Name )
      << (r_Atom_Pure_Name << "Atom_Pure_Name" << apply_Atom_Pure_Name << T_Tilde << T_Name )
      << (r_Atom_String << "Atom_String" << apply_Atom_String << T_String )
      << (r_Atom_RString << "Atom_RString" << apply_Atom_RString << T_RString )
      << (r_Atom_IString << "Atom_IString" << apply_Atom_IString << T_IString )
      << (r_Atom_MString << "Atom_MString" << apply_Atom_MString << T_MString )
      << (r_Atom_False<< "Atom_False" << apply_Atom_False << T_false )
      << (r_Atom_True<< "Atom_True" << apply_Atom_True << T_true )
      << (r_Atom_self<< "Atom_Self" << apply_Atom_Self << T_self )
      << (r_Atom_fself<< "Atom_FSelf" << apply_Atom_FSelf << T_fself )
      << (r_Atom_init<< "Atom_Init" << apply_Atom_Init << T_init )
      << (r_Atom_Nil<< "Atom_Nil" << apply_Atom_Nil << T_nil )
      ;

   ListExpr << "ListExpr";
   ListExpr << ListExpr_errhand;

   ListExpr<< (r_ListExpr_eol << "ListExpr_eol" << apply_dummy << T_EOL );
   ListExpr<< (r_ListExpr_nextd << "ListExpr_nextd" << apply_ListExpr_next2 << ListExpr << T_EOL );

   ListExpr<< (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr );
   ListExpr<< (r_ListExpr_next_no_comma << "ListExpr_next_no_comma" << apply_ListExpr_next_no_comma << ListExpr << Expr );
   ListExpr<< (r_ListExpr_first << "ListExpr_first" << apply_ListExpr_first << Expr );
   ListExpr<< (r_ListExpr_empty << "ListExpr_empty" << apply_ListExpr_empty );

   NeListExpr << "NeListExpr";
   NeListExpr << expr_errhand;
   NeListExpr<< (r_NeListExpr_next << "NeListExpr_next" << apply_NeListExpr_next << NeListExpr << T_Comma << Expr );
   NeListExpr<< (r_NeListExpr_first << "NeListExpr_first" << apply_NeListExpr_first << Expr );
   //NeListExpr.prio(190); // just right below "="

   NeListExpr_ungreed << "NeListExpr_ungreed" << expr_errhand;
   NeListExpr_ungreed<< (r_NeListExpr_ungreed_next << "NeListExpr_ungreed_next" << apply_NeListExpr_ungreed_next << NeListExpr_ungreed << T_Comma << Expr );
   NeListExpr_ungreed<< (r_NeListExpr_ungreed_first << "NeListExpr_ungreed_first" << apply_NeListExpr_ungreed_first << Expr );
   //r_NeListExpr_ungreed_next.setGreedy(false);

   CaseListRange << "CaseListRange";
   CaseListRange << (r_CaseListRange_int << "CaseListRange int" << apply_CaseListRange_int << T_Int << T_to << T_Int );
   CaseListRange << (r_CaseListRange_string << "CaseListRange string" << apply_CaseListRange_string << T_String << T_to << T_String );
   
   CaseListToken << "CaseListToken";
   CaseListToken << (r_CaseListToken_range << "CaseListRange" << apply_CaseListToken_range << CaseListRange );
   CaseListToken << (r_CaseListToken_nil << "CaseListToken nil" << apply_CaseListToken_nil << T_nil );
   CaseListToken << (r_CaseListToken_true << "CaseListToken true" << apply_CaseListToken_true << T_true );
   CaseListToken << (r_CaseListToken_false << "CaseListToken false" << apply_CaseListToken_false << T_false );
   CaseListToken << (r_CaseListToken_int << "CaseListToken int" << apply_CaseListToken_int << T_Int );
   CaseListToken << (r_CaseListToken_string << "CaseListToken string" << apply_CaseListToken_string << T_String );
   CaseListToken << (r_CaseListToken_rstring << "CaseListToken r-string" << apply_CaseListToken_rstring << T_RString );
   CaseListToken << (r_CaseListToken_sym << "CaseListToken sym" << apply_CaseListToken_sym << T_Name );
   
   CaseList << "CaseList";
   CaseList<< (r_CaseList_next << "CaseList_next" << apply_CaseList_next << CaseList << T_Comma << CaseListToken );
   CaseList<< (r_CaseList_first << "CaseList_first" << apply_CaseList_first << CaseListToken );
   // remove right associativity to be able to use "in" in catches.
   // (in is an operator and has a priority, but it is used as a keyword token in catches)
   CaseList.setRightAssoc(true);
   r_CaseList_next.setGreedy(false);

   ListSymbol << "ListSymbol";
   ListSymbol<< (r_ListSymbol_eol << "ListSymbol_eol" << apply_dummy << T_EOL );
   ListSymbol<< (r_ListSymbol_nextd << "ListSymbol_nextd" << apply_ListSymbol_next2 << ListSymbol << T_EOL );

   ListSymbol<< (r_ListSymbol_next << "ListSymbol_next" << apply_ListSymbol_next << ListSymbol << T_Comma << T_Name );
   ListSymbol<< (r_ListSymbol_first << "ListSymbol_first" << apply_ListSymbol_first << T_Name );
   ListSymbol<< (r_ListSymbol_empty << "ListSymbol_empty" << apply_ListSymbol_empty );

   NeListSymbol << "NeListSymbol";
   NeListSymbol<< (r_NeListSymbol_next << "NeListSymbol_next" << apply_NeListSymbol_next << NeListSymbol << T_Comma << T_Name );
   NeListSymbol<< (r_NeListSymbol_first << "NeListSymbol_first" << apply_NeListSymbol_first << T_Name );

   //==================================
   // Class
   S_Class << "Class" << classdecl_errhand;
   S_Class << (r_class << "Class decl" << apply_class_statement << T_class << T_Name );

   S_Object << "Object" << classdecl_errhand;
   S_Object << (r_object << "Object decl" << apply_object_statement << T_object << T_Name );

   FromClause << "Class from clause";
   FromClause << ( r_FromClause_next << "FromClause_next"
              << apply_FromClause_next << FromClause << T_Comma << FromEntry );
   FromClause << ( r_FromClause_first << "FromClause_first" << apply_FromClause_first << FromEntry );

   FromEntry << "Class from entry";
   FromEntry << ( r_FromClause_entry_with_expr << "FromEntry_with_expr"
             << apply_FromClause_entry_with_expr << T_Name << T_Openpar << ListExpr << T_Closepar );
   FromEntry << ( r_FromClause_entry << "FromEntry" << apply_FromClause_entry << T_Name );


   S_PropDecl << "Property declaration";
   S_PropDecl << (r_propdecl_expr << "Expression Property" << apply_pdecl_expr
                               << T_Name << T_EqSign << Expr << T_EOL );

   S_InitDecl << (r_init << "Init block" << apply_init_expr
                               << T_init << T_EOL );

   //==========================================================================
   // Lambdas
   //

   LambdaParams << "LambdaParams";
   LambdaParams << ( r_lit_params << "Params in lit" << apply_lit_params 
                        << T_Openpar << ListSymbol << T_Closepar );
   LambdaParams << ( r_lit_params_eta << "Params in lit ETA" << apply_lit_params_eta 
                        << T_OpenSquare << ListSymbol << T_CloseSquare );
   
   LambdaParams << ( r_lambda_params_eta << "Params in lambda ETA" << apply_lambda_params_eta
                        << ListSymbol << T_STARARROW );
   LambdaParams << ( r_lambda_params << "Params in lambda" << apply_lambda_params 
                        << ListSymbol << T_Arrow );

   EPBody << ( r_lit_epbody << "EP" << apply_ep_body << ListExpr << T_Closepar );

   AccumulatorBody << ( r_accumulator_complete << "Accumulator complete" << apply_accumulator_complete << ListExpr << T_CloseSquare
               << Expr <<  T_Arrow << Expr );
   AccumulatorBody << ( r_accumulator_w_target << "Accumulator target" << apply_accumulator_w_target << ListExpr << T_CloseSquare
               << T_Arrow << Expr );
   AccumulatorBody << ( r_accumulator_w_filter << "Accumulator filter" << apply_accumulator_w_filter << ListExpr << T_CloseSquare
               << Expr );
   AccumulatorBody << ( r_accumulator_simple << "Accumulator simple" << apply_accumulator_simple << ListExpr << T_CloseSquare );

   //==========================================================================
   // Class/Object heading
   //
   ClassParams << "ClassParams" << classdecl_errhand;
   ClassParams << (r_class_from << "Class w/from" << apply_class_from
                  << T_from << FromClause << T_EOL );
   ClassParams << (r_class_pure << "Class decl" << apply_class << T_EOL );
   ClassParams << (r_class_p_from << "AClass w/params & from" << apply_class_p_from
                  << T_Openpar << ListSymbol << T_Closepar << T_from << FromClause << T_EOL );
   ClassParams << (r_class_p << "Class w/params" << apply_class_p
             << T_Openpar << ListSymbol << T_Closepar  << T_EOL );

   // Objects are like classes, but they don't use parameters.
   ObjectParams << "ObjectParams" << classdecl_errhand;
   ObjectParams << r_class_from;
   ObjectParams << r_class_pure;
   
   //==========================================================================
   // prototype
   //
   S_ProtoProp << "S_ProtoProp";
   S_ProtoProp << ( r_proto_prop << "proto_prop" << apply_proto_prop
                        << T_Name << T_EqSign << Expr << T_EOL );

   //==========================================================================
   // Array entries
   //
   ArrayEntry << "ArrayEntry";
   ArrayEntry << ArrayEntry_errHand;
   ArrayEntry << ( r_array_entry_comma << "array_entry_comma" << apply_array_entry_comma << T_Comma );
   ArrayEntry << ( r_array_entry_eol << "array_entry_eol" << apply_array_entry_eol << T_EOL );
   ArrayEntry << ( r_array_entry_arrow << "array_entry_arrow" << apply_array_entry_arrow << T_Arrow );
   ArrayEntry << ( r_array_entry_close << "array_entry_close" << apply_array_entry_close << T_CloseSquare );
   // a little trick; other than being ok, this Non terminal followed by a terminal raises the required arity
   // otherwise, Expr would match early.
   ArrayEntry << ( r_array_entry_range3 << "array_entry_range3" << apply_array_entry_range3 
         << Expr << T_Colon << Expr << T_Colon << Expr << T_CloseSquare );
   ArrayEntry << ( r_array_entry_range3bis << "array_entry_range3bis" << apply_array_entry_range3bis
         << Expr << T_Colon << T_Colon << Expr << T_CloseSquare );
   ArrayEntry << ( r_array_entry_range2 << "array_entry_range2" << apply_array_entry_range2 
         << Expr << T_Colon << Expr << T_CloseSquare );
   ArrayEntry << ( r_array_entry_range1 << "array_entry_range1" << apply_array_entry_range1 
         << Expr << T_Colon << T_CloseSquare );
   ArrayEntry << ( r_array_entry_expr2 << "array_entry_expr2" << apply_array_entry_expr << Expr << T_EOL );
   ArrayEntry << ( r_array_entry_expr1 << "array_entry_expr1" << apply_array_entry_expr << Expr );

   // Handle runaway errors.
   ArrayEntry << (r_array_entry_runaway << "array_entry_runaway" << apply_array_entry_runaway << UnboundKeyword );

   UnboundKeyword << "UnboundKeyword"
                  << (r_uk_if << "UK_if" << T_if )
                  << (r_uk_elif << "UK_elif" << T_elif )
                  << (r_uk_else << "UK_else" << T_else )
                  << (r_uk_while << "UK_while" << T_while )
                  << (r_uk_for << "UK_for" << T_for )
                  //... more to come
                  ;
   
   //==========================================================================
   // Array entries
   //

   //==========================================================================
   //State declarations
   //
   s_Main << "Main"
      << S_EmptyLine
      << S_Load
      << S_Export
      << S_Global
      << S_Import
      << S_Namespace
      
      << S_Function
      << S_Class
      << S_Object
      
      << S_FastPrint
      << S_If
      << S_Elif
      << S_Else
      << S_While
      << S_Loop
      << S_Continue
      << S_Break
      << S_For
      << S_Forfirst
      << S_Formiddle
      << S_Forlast
      << S_Switch
      << S_Case
      << S_Default
      << S_Select
      << S_Try
      << S_Catch
      << S_Finally   
      << S_Raise
      << S_Cut
      << S_Doubt
      << S_RuleOr
      << S_End
      << S_Return
      << S_Autoexpr
      << S_Attribute
      ;

   s_ClassBody << "ClassBody"
      << S_Attribute
      << S_Function
      << S_PropDecl
      << S_InitDecl
      << S_SmallEnd
      << S_EmptyLine
      ;
   

   s_InlineFunc << "InlineFunc"
      << S_Attribute
      << S_EmptyLine
      << S_Global
      << S_FastPrint
      << S_If
      << S_Elif
      << S_Else
      << S_While
      << S_Loop
      << S_Continue
      << S_Break
      << S_For
      << S_Forfirst
      << S_Formiddle
      << S_Forlast
      << S_Switch
      << S_Case
      << S_Default
      << S_Select
      << S_Try
      << S_Catch
      << S_Finally
      << S_Raise
      << S_Cut
      << S_Doubt
      << S_RuleOr
      << S_SmallEnd
      << S_Return
      << S_Autoexpr
      ;

   s_LambdaStart << "LambdaStart"
      << LambdaParams
      << S_EmptyLine
      ;
   
   s_EPState << "EPState"
         << EPBody
         ;

   s_ClassStart << "ClassStart"
      << ClassParams
      ;

   s_ObjectStart << "ObjectStart"
      << ObjectParams
      ;
   
   s_ProtoDecl << "ProtoDecl"
      << S_ProtoProp
      << S_EmptyLine
      << S_SmallEnd
      ;

    s_ArrayDecl << "ArrayDecl"
      << ArrayEntry
      ;

   addState( s_Main );
   addState( s_InlineFunc );
   addState( s_ClassBody );
   addState( s_LambdaStart );
   addState( s_ProtoDecl );
   addState( s_ArrayDecl );
   addState( s_ClassStart );
   addState( s_ObjectStart );
   addState( s_EPState );

}

void SourceParser::onPushState( bool isPushedState )
{
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   pc->onStatePushed( isPushedState );
}


void SourceParser::onPopState()
{
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   pc->onStatePopped();
}

bool SourceParser::parse()
{
   // we need a context (and better to be a SourceContext
   if ( m_ctx == 0 )
   {
      throw new CodeError( ErrorParam( e_setup, __LINE__, SRC ).extra("SourceParser::parse - setContext") );
   }

   bool result =  Parser::parse("Main");
   if( result ) {
      ParserContext* pc = static_cast<ParserContext*>(m_ctx);
      pc->onInputOver();
      result = ! hasErrors();
   }
   return result;
}

void SourceParser::reset()
{
   Parser::reset();
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
   pc->reset();
}


void SourceParser::addError( int code, const String& uri, int l, int c, int ctx, const String& extra )
{
#ifndef NDEBUG
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
#endif
   Parser::addError( code, uri, l, c, ctx, extra );
}

void SourceParser::addError( Error* err )
{
#ifndef NDEBUG
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
#endif
   Parser::addError( err );
}



void SourceParser::addError( int code, const String& uri, int l, int c, int ctx )
{
#ifndef NDEBUG
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
#endif
   Parser::addError( code, uri, l, c, ctx );
}

}

/* end of sourceparser.cpp */
