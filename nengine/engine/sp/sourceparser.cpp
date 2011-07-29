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
#include <falcon/codeerror.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>

#include <falcon/sp/parser_arraydecl.h>
#include <falcon/sp/parser_assign.h>
#include <falcon/sp/parser_atom.h>
#include <falcon/sp/parser_autoexpr.h>
#include <falcon/sp/parser_class.h>
#include <falcon/sp/parser_call.h>
#include <falcon/sp/parser_end.h>
#include <falcon/sp/parser_expr.h>
#include <falcon/sp/parser_function.h>
#include <falcon/sp/parser_if.h>
#include <falcon/sp/parser_index.h>
#include <falcon/sp/parser_list.h>
#include <falcon/sp/parser_proto.h>
#include <falcon/sp/parser_reference.h>
#include <falcon/sp/parser_rule.h>
#include <falcon/sp/parser_while.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

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
   T_Openpar("("),
   T_Closepar(")"),
   T_OpenSquare("["),
   T_DotPar(".("),
   T_DotSquare(".["),
   T_CloseSquare("]"),
   T_OpenGraph("{"),
   T_OpenProto("p{"),
   T_CloseGraph("}"),

   T_Dot("."),
   T_Arrow("=>", 170 ),
   T_AutoAdd( "+=", 70 ),
   T_AutoSub( "-=", 70 ),
   T_AutoTimes( "*=", 70 ),
   T_AutoDiv( "/=", 70 ),
   T_AutoMod( "%=", 70 ),
   T_AutoPow( "**=", 70 ),
   
   T_Comma( "," , 180 ),
   T_Cut("!"),

   T_UnaryMinus("(neg)",23),
   T_Dollar("$",23),
   T_Power("**", 25),

   T_Times("*",30),
   T_Divide("/",30),
   T_Modulo("%",30),

   T_Plus("+",50),
   T_Minus("-",50),
   T_PlusPlus("++",210),

   T_DblEq("==", 70),
   T_NotEq("!=", 70),
   T_Less("<", 70),
   T_Greater(">", 70),
   T_LE("<=", 70),
   T_GE(">=", 70),
   T_Colon( ":" ),
   T_EqSign("=", 200, false),
   T_EqSign2("=", 200 ),


   T_as("as"),
   T_eq("eq", 70 ),
   T_if("if"),
   T_in("in", 20),
   T_or("or", 130),
   T_to("to", 70),

   T_and("and", 120),
   T_def("def"),
   T_end("end"),
   T_for("for"),
   T_not("not", 50),
   T_nil("nil"),
   T_try("try"),

   T_elif("elif"),
   T_else("else"),
   T_rule("rule"),

   T_while("while"),

   T_function("function"),
   T_return("return"),
   T_class("class"),
   T_init("init"),

   T_true( "true" ),
   T_false( "false" ),
   T_self( "self" ),
   T_from( "from" )
{
   S_Autoexpr << "Autoexpr"
      << (r_line_autoexpr << "Autoexpr" << apply_line_expr << Expr << T_EOL)
      << (r_assign_list << "Autoexpr_list" << apply_autoexpr_list << S_MultiAssign << T_EOL )
      ;

   S_If << "IF" << errhand_if;
   S_If << (r_if_short << "if_short" << apply_if_short << T_if << Expr << T_Colon << Expr << T_EOL );
   S_If << (r_if << "if" << apply_if << T_if << Expr << T_EOL );

   S_Elif << "ELIF"
      << (r_elif << "elif" << apply_elif << T_elif << Expr << T_EOL )
      ;

   S_Else << "ELSE"
      << (r_else << "else" << apply_else << T_else << T_EOL )
      ;

   S_While << "WHILE"
      << (r_while_short << "while_short" << apply_while_short << T_while << Expr << T_Colon << Expr << T_EOL )
      << (r_while << "while" << apply_while << T_while << Expr << T_EOL )
      ;

   S_Rule << "RULE"
      << (r_rule << "rule" << apply_rule << T_rule << T_EOL )
      ;

   S_Cut << "CUT"
      << (r_cut << "cut" << apply_cut << T_Cut << T_EOL )
      ;

   S_End << "END"
      << (r_end_rich << "RichEnd" << apply_end_rich << T_end << Expr << T_EOL )
      << (r_end << "end" << apply_end << T_end << T_EOL)
      ;

   S_SmallEnd << "END"
      << (r_end_small << "end_small" << apply_end_small << T_end )
      ;

   S_EmptyLine << "EMPTY"
      << (r_empty << "Empty line" << apply_dummy << T_EOL )
      ;

   S_MultiAssign << "MultiAssign"
      << (r_Stmt_assign_list << "STMT_assign_list" << apply_stmt_assign_list << NeListExpr_ungreed << T_EqSign << NeListExpr )
      ;

  //==========================================================================
  // Expression
  //
  Expr << "Expr";
  Expr << expr_errhand;
  Expr << (r_Expr_assign << "Expr_assign" << apply_expr_assign << Expr << T_EqSign << NeListExpr );

   Expr<< (r_Expr_equal << "Expr_equal" << apply_expr_equal << Expr << T_DblEq << Expr);
   Expr<< (r_Expr_diff << "Expr_diff" << apply_expr_diff << Expr << T_NotEq << Expr);
   Expr<< (r_Expr_less << "Expr_less" << apply_expr_less << Expr << T_Less << Expr);
   Expr<< (r_Expr_greater << "Expr_greater" << apply_expr_greater << Expr << T_Greater << Expr);
   Expr<< (r_Expr_le << "Expr_le" << apply_expr_le << Expr << T_LE << Expr);
   Expr<< (r_Expr_ge << "Expr_ge" << apply_expr_ge << Expr << T_GE << Expr);
   Expr<< (r_Expr_eeq << "Expr_eeq" << apply_expr_eeq << Expr << T_eq << Expr);

   Expr<< (r_Expr_call << "Expr_call" << apply_expr_call << Expr << T_Openpar << ListExpr << T_Closepar );
   Expr<< (r_Expr_index << "Expr_index" << apply_expr_index << Expr << T_OpenSquare << Expr << T_CloseSquare );
   Expr<< (r_Expr_star_index << "Expr_star_index" << apply_expr_star_index << Expr << T_OpenSquare << T_Times << Expr << T_CloseSquare );
   
   Expr<< (r_Expr_array_decl << "Expr_array_decl" << apply_expr_array_decl << T_OpenSquare );
   Expr<< (r_Expr_array_decl2 << "Expr_array_decl2" << apply_expr_array_decl2 << T_DotSquare );

   Expr<< (r_Expr_ref << "Expr_ref" << apply_expr_ref << T_Dollar << T_Name );
   
   Expr<< (r_Expr_dot << "Expr_dot" << apply_expr_dot << Expr << T_Dot << T_Name);
   Expr<< (r_Expr_plus << "Expr_plus" << apply_expr_plus << Expr << T_Plus << Expr);
   Expr<< (r_Expr_preinc << "Expr_preinc" << apply_expr_preinc << T_PlusPlus << Expr);
   Expr<< (r_Expr_postinc << "Expr_postinc" << apply_expr_postinc << Expr << T_PlusPlus);
   Expr<< (r_Expr_minus << "Expr_minus" << apply_expr_minus << Expr << T_Minus << Expr);
   Expr<< (r_Expr_pars << "Expr_pars" << apply_expr_pars << T_Openpar << Expr << T_Closepar);
   Expr<< (r_Expr_pars2 << "Expr_pars2" << apply_expr_pars << T_DotPar << Expr << T_Closepar);
   Expr<< (r_Expr_times << "Expr_times" << apply_expr_times << Expr << T_Times << Expr);
   Expr<< (r_Expr_div   << "Expr_div"   << apply_expr_div   << Expr << T_Divide << Expr );
   Expr<< (r_Expr_pow   << "Expr_pow"   << apply_expr_pow   << Expr << T_Power << Expr );
   Expr<< (r_Expr_auto_add << "Expr_auto_add"   << apply_expr_auto_add   << Expr << T_AutoAdd << Expr );
   Expr<< (r_Expr_auto_sub << "Expr_auto_sub"   << apply_expr_auto_sub   << Expr << T_AutoSub << Expr );
   Expr<< (r_Expr_auto_times << "Expr_auto_times"   << apply_expr_auto_times   << Expr << T_AutoTimes << Expr );
   Expr<< (r_Expr_auto_div << "Expr_auto_div"   << apply_expr_auto_div   << Expr << T_AutoDiv << Expr );
   Expr<< (r_Expr_auto_mod << "Expr_auto_mod"   << apply_expr_auto_mod   << Expr << T_AutoMod << Expr );
   Expr<< (r_Expr_auto_pow << "Expr_auto_pow"   << apply_expr_auto_pow   << Expr << T_AutoPow << Expr );
   // the lexer may find a non-unary minus when parsing it not after an operator...;
   Expr<< (r_Expr_neg   << "Expr_neg"   << apply_expr_neg << T_Minus << Expr );
   // ... or find an unary minus when getting it after another operator.;
   Expr<< (r_Expr_neg2   << "Expr_neg2"   << apply_expr_neg << T_UnaryMinus << Expr );
   Expr<< (r_Expr_Atom << "Expr_atom" << apply_expr_atom << Atom);
   Expr<< (r_Expr_function << "Expr_func" << apply_expr_func << T_function << T_Openpar << ListSymbol << T_Closepar << T_EOL);
   // Start of lambda expressions.
   Expr<< (r_Expr_lambda << "Expr_lambda" << apply_expr_lambda << T_OpenGraph );
   Expr<< (r_Expr_proto << "Expr_proto" << apply_expr_proto << T_OpenProto );

   S_Function << "Function"
      /* This requires a bit of work << (r_function_short << "Function short" << apply_function_short
            << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar <<  T_Colon << Expr << T_EOL )
       */
      << (r_function << "Function decl" << apply_function
             << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL )
      ;
      
   S_Return << "Return"
      << (r_return << "return" << apply_return << T_return << Expr << T_EOL)
      ;

   Atom << "Atom"
      << (r_Atom_Int << "Atom_Int" << apply_Atom_Int << T_Int )
      << (r_Atom_Float << "Atom_Float" << apply_Atom_Float << T_Float )
      << (r_Atom_Name << "Atom_Name" << apply_Atom_Name << T_Name )
      << (r_Atom_String << "Atom_String" << apply_Atom_String << T_String )
      << (r_Atom_False<< "Atom_False" << apply_Atom_False << T_false )
      << (r_Atom_True<< "Atom_True" << apply_Atom_True << T_true )
      << (r_Atom_self<< "Atom_Self" << apply_Atom_Self << T_self )
      << (r_Atom_Nil<< "Atom_Nil" << apply_Atom_Nil << T_nil )
      ;

   ListExpr << "ListExpr";
   ListExpr << ListExpr_errhand;
   ListExpr<< (r_ListExpr_next << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr );
   ListExpr<< (r_ListExpr_first << "ListExpr_first" << apply_ListExpr_first << Expr );
   ListExpr<< (r_ListExpr_empty << "ListExpr_empty" << apply_ListExpr_empty );

   NeListExpr << "NeListExpr";
   NeListExpr << ListExpr_errhand;
   NeListExpr<< (r_NeListExpr_next << "NeListExpr_next" << apply_NeListExpr_next << NeListExpr << T_Comma << Expr );
   NeListExpr<< (r_NeListExpr_first << "NeListExpr_first" << apply_NeListExpr_first << Expr );


   NeListExpr_ungreed << "NeListExpr_ungreed";
   NeListExpr_ungreed << ListExpr_errhand;
   NeListExpr_ungreed<< (r_NeListExpr_ungreed_next << "NeListExpr_ungreed_next" << apply_NeListExpr_ungreed_next << NeListExpr_ungreed << T_Comma << Expr );
   NeListExpr_ungreed<< (r_NeListExpr_ungreed_first << "NeListExpr_ungreed_first" << apply_NeListExpr_ungreed_first << Expr );
   r_NeListExpr_ungreed_next.setGreedy(false);

   ListSymbol << "ListSymbol";
   ListSymbol << ListExpr_errhand;
   ListSymbol<< (r_ListSymbol_next << "ListSymbol_next" << apply_ListSymbol_next << ListSymbol << T_Comma << T_Name );
   ListSymbol<< (r_ListSymbol_first << "ListSymbol_first" << apply_ListSymbol_first << T_Name );
   ListSymbol<< (r_ListSymbol_empty << "ListSymbol_empty" << apply_ListSymbol_empty );

   NeListSymbol << "NeListSymbol";
   NeListSymbol << ListExpr_errhand;
   NeListSymbol<< (r_NeListSymbol_next << "NeListSymbol_next" << apply_NeListSymbol_next << NeListSymbol << T_Comma << T_Name );
   NeListSymbol<< (r_NeListSymbol_first << "NeListSymbol_first" << apply_NeListSymbol_first << T_Name );

   //==================================
   // Class
   S_Class << "Class";
   S_Class << (r_class_from << "Class w/from" << apply_class_from 
               << T_class << T_Name << T_from << FromClause << T_EOL );
   S_Class << (r_class << "Class decl" << apply_class << T_class << T_Name << T_EOL );
   S_Class << (r_class_p_from << "Class w/params & from" << apply_class_p_from
             << T_class << T_Name << T_Openpar << ListSymbol << T_Closepar << T_from << FromClause << T_EOL );
   S_Class << (r_class_p << "Class w/params" << apply_class_p
             << T_class << T_Name << T_Openpar << ListSymbol << T_Closepar  << T_EOL );

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
   LambdaParams << ( r_lambda_params << "Params in lambda" << apply_lambda_params 
                        << ListSymbol << T_Arrow );

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
   ArrayEntry << ( r_array_entry_expr2 << "array_entry_expr2" << apply_array_entry_expr << Expr << T_EOL );
   ArrayEntry << ( r_array_entry_expr1 << "array_entry_expr1" << apply_array_entry_expr << Expr );

   // Handle runaway errors.
   ArrayEntry << (r_array_entry_runaway << "array_entry_runaway" << apply_array_entry_runaway << UnboundKeyword );

   UnboundKeyword << "UnboundKeyword"
                  << (r_uk_if << "UK_if" << T_if )
                  << (r_uk_elif << "UK_elif" << T_elif )
                  << (r_uk_else << "UK_else" << T_else )
                  << (r_uk_while << "UK_while" << T_while )
                  //... more to come
                  ;
   
   //==========================================================================
   // Array entries
   //

   //==========================================================================
   //State declarations
   //
   s_Main << "Main"
      << S_Function
      << S_Class
      << S_Autoexpr
      << S_If
      << S_Elif
      << S_Else
      << S_While
      << S_Rule
      << S_Cut
      << S_End
      << S_Return
      << S_EmptyLine
      ;

   s_ClassBody << "ClassBody"
      << S_Function
      << S_PropDecl
      << S_InitDecl
      << S_End
      << S_EmptyLine
      ;

   s_InlineFunc << "InlineFunc"
      << S_Function
      << S_Class
      << S_Autoexpr
      << S_If
      << S_Elif
      << S_Else
      << S_While
      << S_Rule
      << S_Cut
      << S_SmallEnd
      << S_Return
      << S_EmptyLine
      ;

   s_LambdaStart << "LambdaStart"
      << LambdaParams
      << S_EmptyLine
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

   return Parser::parse("Main");
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
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
   Parser::addError( code, uri, l, c, ctx, extra );
   pc->abandonSymbols();
}


void SourceParser::addError( int code, const String& uri, int l, int c, int ctx )
{
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
   Parser::addError( code, uri, l, c, ctx );
   pc->abandonSymbols();
}


}

/* end of sourceparser.cpp */
