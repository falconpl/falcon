/*
   FALCON - The Falcon Programming Language.
   FILE: dyncompiler.h

   Falcon core module -- Compile function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 11:12:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/dymcompiler.cpp"

#include <falcon/dyncompiler.h>
#include <falcon/syntree.h>
#include <falcon/transcoder.h>
#include <falcon/symbol.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/sourcelexer.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/syntree.h>
#include <falcon/module.h>
#include <falcon/attribute_helper.h>
#include <falcon/stringstream.h>
#include <falcon/falconclass.h>
#include <falcon/stderrors.h>







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
#include <falcon/parser/lexer.h>
#include <falcon/parser/parser.h>



namespace Falcon {

//=========================================================================================
//=========================================================================================
//=========================================================================================
//=========================================================================================
//=========================================================================================
using namespace Parsing;

/** Class reading a Falcon script source.
 */
class FALCON_DYN_CLASS LSourceParser: public Parsing::Parser
{
public:
   LSourceParser();
   virtual ~LSourceParser();
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
   Parsing::NonTerminal S_Attribute;
   Parsing::NonTerminal S_Autoexpr;
   Parsing::NonTerminal S_If;

   Parsing::NonTerminal S_Elif;
   Parsing::NonTerminal S_Else;
   Parsing::NonTerminal S_While;
   Parsing::NonTerminal S_Continue;
   Parsing::NonTerminal S_Break;
   Parsing::NonTerminal S_For;
   Parsing::NonTerminal S_Forfirst;
   Parsing::NonTerminal S_Formiddle;
   Parsing::NonTerminal S_Forlast;
   Parsing::NonTerminal S_Switch;
   Parsing::NonTerminal S_Select;
   Parsing::NonTerminal S_Case;
   Parsing::NonTerminal S_Default;
   Parsing::NonTerminal S_RuleOr;
   Parsing::NonTerminal S_Cut;
   Parsing::NonTerminal S_Doubt;
   Parsing::NonTerminal S_End;
   Parsing::NonTerminal S_SmallEnd;
   Parsing::NonTerminal S_EmptyLine;
   Parsing::NonTerminal S_MultiAssign;
   Parsing::NonTerminal S_FastPrint;
   Parsing::NonTerminal PrintExpr;

   //================================================
   // Load, import and export
   //
   Parsing::NonTerminal S_Load;
   Parsing::NonTerminal ModSpec;
   Parsing::NonTerminal S_Export;
   Parsing::NonTerminal S_Global;

   Parsing::NonTerminal S_Try;
   Parsing::NonTerminal S_Catch;
   Parsing::NonTerminal S_Finally;
   Parsing::NonTerminal CatchSpec;
   Parsing::NonTerminal S_Raise;

   Parsing::NonTerminal S_Import;
   Parsing::NonTerminal ImportClause;
   Parsing::NonTerminal ImportSpec;
   Parsing::NonTerminal NameSpaceSpec;

   Parsing::NonTerminal S_Loop;

   Parsing::NonTerminal S_Namespace;

   Parsing::NonTerminal Expr;

   Parsing::NonTerminal S_Function;
   Parsing::NonTerminal S_Return;
   Parsing::NonTerminal S_Class;
   Parsing::NonTerminal S_Object;
   Parsing::NonTerminal S_InitDecl;
   Parsing::NonTerminal FromClause;
   Parsing::NonTerminal FromEntry;
   Parsing::NonTerminal S_PropDecl;
   Parsing::NonTerminal S_StaticPropDecl;

   Parsing::NonTerminal Atom;

   //================================================
   // Expression lists
   //
   Parsing::NonTerminal ListExpr;
   Parsing::NonTerminal CaseListRange;
   Parsing::NonTerminal CaseListToken;
   Parsing::NonTerminal CaseList;
   Parsing::NonTerminal NeListExpr;
   Parsing::NonTerminal NeListExpr_ungreed;

   //================================================
   // Symbol list
   //

   Parsing::NonTerminal ListSymbol;
   Parsing::NonTerminal NeListSymbol;
   Parsing::NonTerminal LambdaParams;
   Parsing::NonTerminal EPBody;
   Parsing::NonTerminal AccumulatorBody;
   Parsing::NonTerminal ClassParams;
   Parsing::NonTerminal ObjectParams;
   Parsing::NonTerminal S_ProtoProp;
   Parsing::NonTerminal ArrayEntry;
   Parsing::NonTerminal UnboundKeyword;

   //=======================================================
   // States
   //
   Parsing::NonTerminal MainProgram;
   Parsing::NonTerminal ClassBody;
   Parsing::NonTerminal InlineFunc;
   Parsing::NonTerminal LambdaStart;
   Parsing::NonTerminal EPState;
   Parsing::NonTerminal ClassStart;
   Parsing::NonTerminal ObjectStart;
   Parsing::NonTerminal ProtoDecl;
   Parsing::NonTerminal ArrayDecl;

private:
   void init();

};


static void apply_dummy( const NonTerminal&, Parser&p )
{
   p.simplify(1);
}

//==========================================================
// SourceParser
//==========================================================

LSourceParser::LSourceParser():
   T_Openpar("'('",20),
   T_Closepar("')'"),
   T_OpenSquare("'['", 20),
   T_CapPar("'^('"),
   T_CapSquare("'^['"),
   T_DotPar("'.('"),
   T_DotSquare("'.['"),
   T_CloseSquare("']'"),
   T_OpenGraph("'{'",20),
   T_OpenProto("'p{'"),
   T_CloseGraph("'}'"),

   T_Dot("'.'",15),
   T_Arrow("'=>'", 170 ),
   T_AutoAdd( "'+='", 70 ),
   T_AutoSub( "'-='", 70 ),
   T_AutoTimes( "'*='", 70 ),
   T_AutoDiv( "'/='", 70 ),
   T_AutoMod( "'%='", 70 ),
   T_AutoPow( "'**='", 70 ),
   T_AutoRShift( "'>>='", 70 ),
   T_AutoLShift( "'<<='", 70 ),
   T_EEQ( "'==='", 70 ),

   T_BAND("'^&'", 60),
   T_BOR("'^|'", 65),
   T_BXOR("'^^'", 65),
   T_BNOT("'^!'", 23),

   T_OOB("'^+'", 24),
   T_DEOOB("'^-'", 24),
   T_XOOB("'^%'", 24),
   T_ISOOB("'^$'", 24),
   T_UNQUOTE("'^~'", 10 ),
   T_COMPOSE("'^.'", 60),
   T_EVALRET( "'^='", 150),
   T_EVALRET_EXEC( "'^*'", 150),
   T_EVALRET_DOUBT( "'^?'", 150),
   T_ETAARROW( "'*=>'", 170),

   T_Comma( "','" , 180 ),
   T_QMark( "'?'" , 175, true ),
   T_Tilde( "'~'" , 5 ),
   T_Bang("'!'"),
   T_Disjunct( "'|'" , 130 ),

   T_UnaryMinus("Unary-minus",23),
   T_Dollar("'$'",23),
   T_Amper("'&'",10),
   T_NumberSign("'#'", 68),   /* between bit and & equality */
   T_At("'@'", 10),           /* Must have a very high priority,
                               as @"abc".x should be interpreted as (@"abc").x */

   T_Power("'**'", 25),

   T_Times("'*'",30),
   T_Divide("'/'",30),
   T_Modulo("'%'",30),
   T_RShift("'>>'",30),
   T_LShift("'<<'",30),

   T_Plus("'+'",50),
   T_Minus("'-'",50),
   T_PlusPlus("'++'",21, true),
   T_MinusMinus("'--'",21, true),

   T_DblEq("'=='", 70),
   T_NotEq("'!='", 70),
   T_Less("'<'", 70),
   T_Greater("'>'", 70),
   T_LE("'<='", 70),
   T_GE("'>='", 70),
   T_Colon( "':'", 170 ),
   T_EqSign("'='", 200, true),
   T_EqSign2("'='", 200 ),


   T_as("'as'"),
   T_if("'if'"),
   T_in("'in'", 23),
   T_or("'or'", 130),
   T_to("'to'", 70),

   T_and("'and'", 120),
   T_def("'def'"),
   T_end("'end'"),
   T_for("'for'"),
   T_not("'not'", 50),
   T_nil("'nil'"),
   T_try("'try'"),
   T_catch("'catch'"),
   T_notin("'notin'", 23),
   T_finally("'finally'"),
   T_raise("'raise'"),
   T_fself("'fself'"),

   T_elif("'elif'"),
   T_else("'else'"),
   T_rule("'rule'"),

   T_while("'while'"),

   T_function("'function'"),
   T_return("'return'"),
   T_class("'class'"),
   T_object("'object'"),
   T_init("'init'"),

   T_true( "'true'" ),
   T_false( "'false'" ),
   T_self( "'self'" ),
   T_from( "'from'" ),
   T_load( "'load'" ),
   T_export( "'export'" ),
   T_import( "'import'" ),
   T_namespace( "'namespace'" ),
   T_global("'global'"),

   T_forfirst( "'forfirst'" ),
   T_formiddle( "'formiddle'" ),
   T_forlast( "'forlast'" ),
   T_break( "'break'" ),
   T_continue( "'continue'" ),

   T_switch("'switch'"),
   T_case("'case'"),
   T_default("'default'"),
   T_select("'select'"),
   T_loop("'loop'"),
   T_static("'static'"),

   T_RString("R-String"),
   T_IString("I-String"),
   T_MString("M-String"),
   T_provides("'provides'"),

   T_DoubleColon("'::'",16),
   T_ColonQMark("':?'",16)
{
   init();
}

void LSourceParser::init()
{
//========================================================================
// Topmost part
//
   MainProgram << "Main"
           << NonTerminal::nr << S_EmptyLine
           << NonTerminal::nr << S_Class
           << NonTerminal::nr << S_Autoexpr
           << NonTerminal::nr << S_FastPrint
           << NonTerminal::nr << S_End
           << NonTerminal::nr << S_Function
           << NonTerminal::nr << S_Return
           << NonTerminal::nr << S_Attribute
           << NonTerminal::nr << S_If
           << NonTerminal::nr << S_Elif
           << NonTerminal::nr << S_Else
           << NonTerminal::nr << S_While
           << NonTerminal::nr << S_Continue
           << NonTerminal::nr << S_Break
           << NonTerminal::nr << S_Loop
           << NonTerminal::nr << S_For
           << NonTerminal::nr << S_Forfirst
           << NonTerminal::nr << S_Formiddle
           << NonTerminal::nr << S_Forlast

           << NonTerminal::nr << S_Object

           << NonTerminal::nr << S_Switch
           << NonTerminal::nr << S_Select
           << NonTerminal::nr << S_Case
           << NonTerminal::nr << S_Default

           << NonTerminal::nr << S_Try
           << NonTerminal::nr << S_Catch
           << NonTerminal::nr << S_Finally
           << NonTerminal::nr << S_Raise

           << NonTerminal::nr << S_Cut
           << NonTerminal::nr << S_Doubt
           << NonTerminal::nr << S_RuleOr

           << NonTerminal::nr << S_Load
           << NonTerminal::nr << S_Import
           << NonTerminal::nr << S_Export
           << NonTerminal::nr << S_Namespace
           << NonTerminal::nr << S_Global

           << NonTerminal::endr;


   ClassBody << "ClassBody"
            << NonTerminal::nr << S_Attribute
            << NonTerminal::nr << S_Function
            << NonTerminal::nr << S_PropDecl
            << NonTerminal::nr << S_InitDecl
            << NonTerminal::nr << S_SmallEnd
            << NonTerminal::nr << S_EmptyLine
            << NonTerminal::nr << S_StaticPropDecl
            << NonTerminal::endr;


   InlineFunc << "InlineFunc"
         << NonTerminal::nr << S_EmptyLine
         << NonTerminal::nr << S_Autoexpr
         << NonTerminal::nr << S_SmallEnd
         << NonTerminal::nr << S_FastPrint
         << NonTerminal::nr << S_If
         << NonTerminal::nr << S_Elif
         << NonTerminal::nr << S_Else
         << NonTerminal::nr << S_Return
         << NonTerminal::nr << S_While
         << NonTerminal::nr << S_Loop
         << NonTerminal::nr << S_Continue
         << NonTerminal::nr << S_Break
         << NonTerminal::nr << S_For
         << NonTerminal::nr << S_Forfirst
         << NonTerminal::nr << S_Formiddle
         << NonTerminal::nr << S_Forlast
         << NonTerminal::nr << S_Switch
         << NonTerminal::nr << S_Case
         << NonTerminal::nr << S_Default
         << NonTerminal::nr << S_Select
         << NonTerminal::nr << S_Try
         << NonTerminal::nr << S_Catch
         << NonTerminal::nr << S_Finally
         << NonTerminal::nr << S_Raise
         << NonTerminal::nr << S_Cut
         << NonTerminal::nr << S_Doubt
         << NonTerminal::nr << S_RuleOr
         << NonTerminal::nr << S_Attribute
         << NonTerminal::nr << S_Global
         << NonTerminal::endr;

   LambdaStart << "LambdaStart"
            << NonTerminal::nr << LambdaParams
            << NonTerminal::nr << S_EmptyLine
            << NonTerminal::endr;

   EPState << "EPState"
            << NonTerminal::nr << EPBody
            << NonTerminal::endr;

   ClassStart << "ClassStart"
            << NonTerminal::nr << ClassParams
            << NonTerminal::endr;

   ObjectStart << "ObjectStart"
            << NonTerminal::nr << ObjectParams
            << NonTerminal::endr;

   ProtoDecl << "ProtoDecl"
            << NonTerminal::nr << S_ProtoProp
            << NonTerminal::nr << S_EmptyLine
            << NonTerminal::nr << S_SmallEnd
            << NonTerminal::endr;

   ArrayDecl << "ArrayDecl"
         << NonTerminal::nr << ArrayEntry
         << NonTerminal::endr;


//=============================================================================
// Expression
//

   Expr << "Expr" << expr_errhand;
   Expr << NonTerminal::sr
          // we need to have named parameters before atom, because it's the only expr
          // that might start with an atom (a name)
         << "named-param" << apply_expr_named << T_Name << T_Disjunct << Expr
         << "atom" << apply_expr_atom << Atom
         << "neg"  << apply_expr_neg << T_Minus << Expr
         << "neg2" << apply_expr_neg << T_UnaryMinus << Expr
         << "not"  << apply_expr_not  << T_not << Expr
         << "Bnot" << apply_expr_bnot << T_BNOT << Expr
         << "oob" << apply_expr_oob << T_OOB << Expr
         << "deoob" << apply_expr_deoob << T_DEOOB << Expr
         << "xoob" << apply_expr_xoob << T_XOOB << Expr
         << "isoob" << apply_expr_isoob << T_ISOOB << Expr
         << "str_ipol" << apply_expr_str_ipol << T_At << Expr

         << "evarlet"  << apply_expr_evalret << T_EVALRET << Expr
         << "evarlet_exec"  << apply_expr_evalret_exec << T_EVALRET_EXEC << Expr
         << "evarlet_doubt"  << apply_expr_evalret_doubt << T_EVALRET_DOUBT << Expr

         << "provides" << apply_expr_provides << Expr << T_provides << T_Name

         << "assign" << apply_expr_assign << Expr << T_EqSign << Expr

         << "equal" << apply_expr_equal << Expr << T_DblEq << Expr
         << "diff" << apply_expr_diff << Expr << T_NotEq << Expr
         << "less" << apply_expr_less << Expr << T_Less << Expr
         << "greater" << apply_expr_greater << Expr << T_Greater << Expr
         << "le" << apply_expr_le << Expr << T_LE << Expr
         << "ge" << apply_expr_ge << Expr << T_GE << Expr
         << "eeq" << apply_expr_eeq << Expr << T_EEQ << Expr
         << "in" << apply_expr_in << Expr << T_in << Expr
         << "notin" << apply_expr_notin << Expr << T_notin << Expr

         << "call" << apply_expr_call << Expr << T_Openpar << ListExpr << T_Closepar
         << "summon" << apply_expr_summon << Expr << T_DoubleColon << T_Name << T_OpenSquare << ListExpr << T_CloseSquare
         << "opt_summon" << apply_expr_opt_summon << Expr << T_ColonQMark << T_Name << T_OpenSquare <<  ListExpr << T_CloseSquare

         << "index" << apply_expr_index << Expr << T_OpenSquare << Expr << T_CloseSquare
         << "star_index" << apply_expr_star_index << Expr << T_OpenSquare << T_Times << Expr << T_CloseSquare
         << "Expr_range_index3" << apply_expr_range_index3
                                     << Expr << T_OpenSquare << Expr << T_Colon << Expr << T_Colon << Expr << T_CloseSquare
         << "Expr_range_index3open" << apply_expr_range_index3open
                                     << Expr << T_OpenSquare << Expr << T_Colon << T_Colon << Expr << T_CloseSquare
         << "Expr_range_index2" << apply_expr_range_index2
                                     << Expr << T_OpenSquare << Expr << T_Colon << Expr << T_CloseSquare
         << "Expr_range_index1" << apply_expr_range_index1
                                     << Expr << T_OpenSquare << Expr << T_Colon << T_CloseSquare
         << "Expr_range_index0" << apply_expr_range_index0
                                     << Expr << T_OpenSquare << T_Colon << T_CloseSquare

         << "array_decl" << apply_expr_array_decl << T_OpenSquare
         << "array_decl2" << apply_expr_array_decl2 << T_DotSquare
         << "accumulator" << apply_expr_accumulator << T_CapSquare << AccumulatorBody

         << "dyns" << apply_expr_symname << T_Dollar << T_Name

         << "dot" << apply_expr_dot << Expr << T_Dot << T_Name
         << "plus" << apply_expr_plus << Expr << T_Plus << Expr
         << "preinc" << apply_expr_preinc << T_PlusPlus << Expr
         << "postinc" << apply_expr_postinc << Expr << T_PlusPlus
         << "predec" << apply_expr_predec << T_MinusMinus << Expr
         << "postdec" << apply_expr_postdec << Expr << T_MinusMinus

         << "minus" << apply_expr_minus << Expr << T_Minus << Expr
         << "pars" << apply_expr_pars << T_Openpar << Expr << T_Closepar
         << "pars2" << apply_expr_pars << T_DotPar << Expr << T_Closepar
         << "times" << apply_expr_times << Expr << T_Times << Expr
         << "div"   << apply_expr_div   << Expr << T_Divide << Expr
         << "mod"   << apply_expr_mod   << Expr << T_Modulo << Expr
         << "pow"   << apply_expr_pow   << Expr << T_Power << Expr
         << "shr"   << apply_expr_shr   << Expr << T_RShift << Expr
         << "shl"   << apply_expr_shl   << Expr << T_LShift << Expr

         << "and" << apply_expr_and  << Expr << T_and << Expr
         << "or"  << apply_expr_or   << Expr << T_or << Expr

         << "band" << apply_expr_band  << Expr << T_BAND << Expr
         << "bor"  << apply_expr_bor   << Expr << T_BOR << Expr
         << "bxor" << apply_expr_bxor  << Expr << T_BXOR << Expr

         << "auto_add"   << apply_expr_auto_add   << Expr << T_AutoAdd << Expr
         << "auto_sub"   << apply_expr_auto_sub   << Expr << T_AutoSub << Expr
         << "auto_times"   << apply_expr_auto_times   << Expr << T_AutoTimes << Expr
         << "auto_div"   << apply_expr_auto_div   << Expr << T_AutoDiv << Expr
         << "auto_mod"   << apply_expr_auto_mod   << Expr << T_AutoMod << Expr
         << "auto_pow"   << apply_expr_auto_pow   << Expr << T_AutoPow << Expr
         << "auto_shr"   << apply_expr_auto_shr  << Expr << T_AutoRShift << Expr
         << "auto_shl"   << apply_expr_auto_shl   << Expr << T_AutoLShift << Expr

         << "invoke"   << apply_expr_invoke   << Expr << T_NumberSign << Expr
         << "compose"  << apply_expr_compose << Expr << T_COMPOSE << Expr

         << "ternary_if"  << apply_expr_ternary_if << Expr << T_QMark << Expr << T_Colon << Expr

         << "unquote"  << apply_expr_unquote << T_UNQUOTE << Expr

         << "func" << apply_expr_func << T_function << T_Openpar << ListSymbol << T_Closepar << T_EOL
         << "funcEta" << apply_expr_funcEta << T_function << T_Amper << T_Openpar << ListSymbol << T_Closepar << T_EOL

         // Start of lambda expressions.
         << "lambda" << apply_expr_lambda << T_OpenGraph
         << "ep" << apply_expr_ep << T_CapPar
         << "class" << apply_expr_class << T_class
         << "proto" << apply_expr_proto << T_OpenProto
         << "rule decl" << apply_rule << T_rule << T_EOL

      << NonTerminal::endr;


   Atom << "Atom"
      << NonTerminal::sr
      << "Atom_Int" << apply_Atom_Int << T_Int
      << "Atom_Float" << apply_Atom_Float << T_Float
      << "Atom_Name" << apply_Atom_Name << T_Name
      << "Atom_Pure_Name" << apply_Atom_Pure_Name << T_Tilde << T_Name
      << "Atom_String" << apply_Atom_String << T_String
      << "Atom_RString" << apply_Atom_RString << T_RString
      << "Atom_IString" << apply_Atom_IString << T_IString
      << "Atom_MString" << apply_Atom_MString << T_MString
      << "Atom_False" << apply_Atom_False << T_false
      << "Atom_True" << apply_Atom_True << T_true
      << "Atom_Self" << apply_Atom_Self << T_self
      << "Atom_FSelf" << apply_Atom_FSelf << T_fself
      << "Atom_Init" << apply_Atom_Init << T_init
      << "Atom_Nil" << apply_Atom_Nil << T_nil
      << NonTerminal::endr
      ;

//====================================================================================
// Middle-sized entities.
//
   ListExpr << "ListExpr" << ListExpr_errhand
         << NonTerminal::sr
         << "ListExpr_eol" << apply_dummy << T_EOL
         << "ListExpr_nextd" << apply_ListExpr_next2 << ListExpr << T_EOL
         << "ListExpr_next" << apply_ListExpr_next << ListExpr << T_Comma << Expr
         << "ListExpr_next_no_comma" << apply_ListExpr_next_no_comma << ListExpr << Expr
         << "ListExpr_first" << apply_ListExpr_first << Expr
         << "ListExpr_empty" << apply_ListExpr_empty
         << NonTerminal::endr;

   NeListExpr << "NeListExpr" << expr_errhand
         << NonTerminal::sr
         // a little trick: this rule succeds, but doesn't remove the '='
         << "next" << apply_NeListExpr_assign << NeListExpr << T_EqSign
         // this will match, but reduce leaving the equal sign.
         << "next" << apply_NeListExpr_next << NeListExpr << T_Comma << Expr << T_EqSign
         << "next" << apply_NeListExpr_next << NeListExpr << T_Comma << Expr
         << "first" << apply_NeListExpr_first << Expr << T_EqSign
         << "first" << apply_NeListExpr_first << Expr
         << NonTerminal::endr;

   ListSymbol << "ListSymbol"
      << NonTerminal::sr
      << "ls-eol" << apply_dummy << T_EOL
      << "ls-nextd" << apply_ListSymbol_next2 << ListSymbol << T_EOL
      << "ls-next" << apply_ListSymbol_next << ListSymbol << T_Comma << T_Name
      << "ls-first" << apply_ListSymbol_first << T_Name
      << "ls-empty" << apply_ListSymbol_empty
      << NonTerminal::endr;

   NeListSymbol << "NeListSymbol"
      << NonTerminal::sr
      << "next" << apply_NeListSymbol_next << NeListSymbol << T_Comma << T_Name
      << "first" << apply_NeListSymbol_first << T_Name
      << NonTerminal::endr;

   LambdaParams << "LambdaParams"
      << NonTerminal::sr
      << "params" << apply_lit_params << T_Openpar << ListSymbol << T_Closepar
      << "ETA-params" << apply_lit_params_eta << T_OpenSquare << ListSymbol << T_CloseSquare
      << "l-ETA-params" << apply_lambda_params_eta << ListSymbol << T_ETAARROW
      << "l-params" << apply_lambda_params << ListSymbol << T_Arrow
      << NonTerminal::endr;

   EPBody << "EPBody"
      << NonTerminal::sr
      << apply_ep_body << ListExpr << T_Closepar
      << NonTerminal::endr;

   AccumulatorBody << "Accumulator-body"
      << NonTerminal::sr
      << "w/all" << apply_accumulator_complete << ListExpr << T_CloseSquare << Expr <<  T_Arrow << Expr
      << "w/target" << apply_accumulator_w_target << ListExpr << T_CloseSquare << T_Arrow << Expr
      << "w/filter" << apply_accumulator_w_filter << ListExpr << T_CloseSquare << Expr
      << "w/o all"  << apply_accumulator_simple << ListExpr << T_CloseSquare
      << NonTerminal::endr;

//==========================================================================
// Array entries
//
   ArrayEntry << "ArrayEntry" << ArrayEntry_errHand
      << NonTerminal::sr
      << "comma" << apply_array_entry_comma << T_Comma
      << "eol" << apply_array_entry_eol << T_EOL
      << "arrow" << apply_array_entry_arrow << T_Arrow
      << "close" << apply_array_entry_close << T_CloseSquare
      // a little trick; other than being ok, this Non terminal followed by a terminal raises the required arity
      // otherwise, Expr would match early.
      << "range3" << apply_array_entry_range3  << Expr << T_Colon << Expr << T_Colon << Expr << T_CloseSquare
      << "range3bis" << apply_array_entry_range3bis << Expr << T_Colon << T_Colon << Expr << T_CloseSquare
      << "range2" << apply_array_entry_range2 << Expr << T_Colon << Expr << T_CloseSquare
      << "range1" << apply_array_entry_range1 << Expr << T_Colon << T_CloseSquare
      << "expr2" << apply_array_entry_expr << Expr << T_EOL
      << "expr1" << apply_array_entry_expr << Expr
      // Handle runaway errors.
      << "runaway" << apply_array_entry_runaway << UnboundKeyword
      << NonTerminal::endr;

   UnboundKeyword << "UnboundKeyword"
                     << NonTerminal::nr << T_if
                     << NonTerminal::nr << T_elif
                     << NonTerminal::nr << T_else
                     << NonTerminal::nr << T_while
                     << NonTerminal::nr << T_for
                     //... more to come
                     << NonTerminal::endr;

//====================================================================================
// General statements
//
   S_EmptyLine << "Empty-Line"
         << NonTerminal::sr << "end-of-line" << T_EOL << NonTerminal::endr;

   S_Autoexpr << "Auto-Expr"
            << NonTerminal::sr
            << "auto-expr" << apply_line_expr << Expr << T_EOL
            << "Autoexpr-list" << apply_autoexpr_list << S_MultiAssign
            << NonTerminal::endr;

   S_MultiAssign << "MultiAssign"
            << NonTerminal::sr
            << "list-assign" << apply_stmt_assign_list << NeListExpr << T_EqSign << NeListExpr << T_EOL
            << NonTerminal::endr;

   S_Global << "Stmt-global"
            << NonTerminal::sr << apply_global << T_global << ListSymbol << T_EOL << NonTerminal::endr;

   S_End << "Stmt-end"
         << NonTerminal::sr
         << "Rich-End" << apply_end_rich << T_end << Expr << T_EOL
         << "end" << apply_end << T_end << T_EOL
         << NonTerminal::endr;

   S_Return << "Stmt-return"
      << NonTerminal::sr
      << "return w/doubt" << apply_return_doubt << T_return << T_QMark << Expr << T_EOL
      << "return w/eval" << apply_return_eval << T_return << T_Amper << Expr << T_EOL
      << "return break" << apply_return_break << T_return << T_break << T_EOL
      << "return expr" << apply_return_expr << T_return << Expr << T_EOL
      << "return" << apply_return << T_return << T_EOL
      << NonTerminal::endr;
      ;

   S_SmallEnd << "Small-END"
         << NonTerminal::sr
         << "end_small" << apply_end_small << T_end
         << NonTerminal::endr;

   S_FastPrint << "Stmt-fastprint" << PrintExpr_errhand
      << NonTerminal::sr
      << "fprint alone" << apply_fastprint_alone << T_RShift << T_EOL
      << "fprint" << apply_fastprint << T_RShift << ListExpr << T_EOL
      << "fastprint+nl alone" << apply_fastprint_nl_alone << T_Greater << T_EOL
      << "fastprint+nl" << apply_fastprint_nl << T_Greater << ListExpr << T_EOL
      << NonTerminal::endr;

   S_Attribute << "Stmt-attribute" << errhand_attribute
      << NonTerminal::sr
      <<"decl" << apply_attribute << T_Colon << T_Name <<  T_Arrow << Expr << T_EOL
      << NonTerminal::endr;

//====================================================================================
// If statement
//

   S_If << "Stmt-if" << errhand_if
        << NonTerminal::sr
        << "short-decl" << apply_if_short << T_if << Expr << T_Colon
        << "decl" << apply_if << T_if << Expr << T_EOL
        << NonTerminal::endr;

   S_Elif << "Stmt-elif"
         << NonTerminal::sr
         << "elif" << apply_elif << T_elif << Expr << T_EOL
         << NonTerminal::endr;

   S_Else << "Stmt-else"
          << NonTerminal::sr
          << "decl" << apply_else << T_else << T_EOL
          << NonTerminal::endr;

//====================================================================================
// Basic loops
//

   S_While << "Stmt-while" << while_errhand
           << NonTerminal::sr
           << "short" << apply_while_short << T_while << Expr << T_Colon
           << "full" << apply_while << T_while << Expr << T_EOL
           << NonTerminal::endr;

   S_Loop << "Stmt-loop" << loop_errhand
           << NonTerminal::sr
           << "short" << apply_loop_short << T_loop << T_Colon
           << "full" << apply_loop << T_loop << T_EOL
           << NonTerminal::endr;

   S_Continue << "Continue"
      << NonTerminal::sr
      << apply_continue << T_continue << T_EOL
      << NonTerminal::endr;

   S_Break << "Break"
      << NonTerminal::sr
      << apply_break << T_break << T_EOL
      << NonTerminal::endr;

//====================================================================================
// For statements
//

   S_For << "Stmt-for" << for_errhand
       << NonTerminal::sr
       << "for-to-step-l" << apply_for_to_step << T_for << T_Name << T_EqSign << Expr << T_to << Expr << T_Comma << Expr << T_EOL
       << "for-to-step-s" << apply_for_to_step_short << T_for << T_Name << T_EqSign << Expr << T_to << Expr << T_Comma << Expr << T_Colon
       << "for-to-l" << apply_for_to << T_for << T_Name << T_EqSign << Expr << T_to << Expr << T_EOL
       << "for-to-s" << apply_for_to_short << T_for << T_Name << T_EqSign << Expr << T_to << Expr << T_Colon
       << "for-in-l" << apply_for_in << T_for << NeListSymbol << T_in << Expr << T_EOL
       << "for-in-s" << apply_for_in_short << T_for << NeListSymbol << T_in << Expr << T_Colon
       << NonTerminal::endr;

   S_Forfirst << "Stmt-forfirst"
       << NonTerminal::sr
       << "long" << apply_forfirst << T_forfirst << T_EOL
       << "short" << apply_forfirst_short << T_forfirst << T_Colon
       << NonTerminal::endr;

   S_Formiddle << "Stmt-formiddle"
       << NonTerminal::sr
       << "long" << apply_formiddle << T_formiddle << T_EOL
       << "short" << apply_formiddle_short << T_formiddle << T_Colon
       << NonTerminal::endr;

   S_Forlast << "Stmt-forlast"
       << NonTerminal::sr
       << "long" << apply_forlast << T_forlast << T_EOL
       << "short" << apply_forlast_short << T_forlast << T_Colon
       << NonTerminal::endr;

//====================================================================================
// Rules
//

   S_Cut << "Stmt-cut"
      << NonTerminal::sr
      << "cut-expr" << apply_cut_expr << T_Bang << Expr << T_EOL
      << "cut" << apply_cut << T_Bang << T_EOL
      << NonTerminal::endr
      ;

   S_Doubt << "Doubt"
      << NonTerminal::sr
      << "doubt-expr" << apply_doubt << T_QMark << Expr << T_EOL
      << NonTerminal::endr
      ;

   S_RuleOr << "Stmt-rule-or"
      << NonTerminal::sr
      << apply_rule_branch << T_or << T_EOL
      << NonTerminal::endr;

//========================================================================================
// Switch
//

   S_Switch << "switch"
      << NonTerminal::sr
      << apply_switch << T_switch << Expr << T_EOL
      << NonTerminal::endr;

   S_Select << "select"
      << NonTerminal::sr
      << apply_select << T_select << Expr << T_EOL
      << NonTerminal::endr;

   S_Case << "case"
      << NonTerminal::sr
      << "long" << apply_case << T_case << CaseList << T_EOL
      << "short" << apply_case_short << T_case << CaseList << T_Colon
      << NonTerminal::endr;

   S_Default << "default"
      << NonTerminal::sr
      << "long" << apply_default << T_default << T_EOL
      << "short" << apply_default_short << T_default << T_Colon
      << NonTerminal::endr;

   CaseListRange << "CaseListRange"
      << NonTerminal::sr
      << "int" << apply_CaseListRange_int << T_Int << T_to << T_Int
      << "string" << apply_CaseListRange_string << T_String << T_to << T_String
      << NonTerminal::endr;

   CaseListToken << "CaseListToken"
      << NonTerminal::sr
      << "range" << apply_CaseListToken_range << CaseListRange
      << "nil" << apply_CaseListToken_nil << T_nil
      << "true" << apply_CaseListToken_true << T_true
      << "false" << apply_CaseListToken_false << T_false
      << "int" << apply_CaseListToken_int << T_Int
      << "string" << apply_CaseListToken_string << T_String
      << "r-string" << apply_CaseListToken_rstring << T_RString
      << "sym" << apply_CaseListToken_sym << T_Name
      << NonTerminal::endr;

   CaseList << "CaseList"
      << NonTerminal::sr
      << "next" << apply_CaseList_next << CaseList << T_Comma << CaseListToken
      << "first" << apply_CaseList_first << CaseListToken
      << NonTerminal::endr;

    // remove right associativity to be able to use "in" in catches.
    // (in is an operator and has a priority, but it is used as a keyword token in catches)
   CaseList.setRightAssoc(true);


//========================================================================================
// Try-catch
//
   S_Try << "Stmt-try" << try_errhand
      << NonTerminal::sr
      << "try_rule" << apply_try << T_try << T_EOL
      << NonTerminal::endr;

   S_Catch << "Stmt-catch" << catch_errhand
      << NonTerminal::sr
      << "aimplw" << apply_catch << T_catch << CatchSpec
      << NonTerminal::endr;

   S_Finally << "Stmt-finally" << finally_errhand
      << NonTerminal::sr
      << "r_finally" << apply_finally << T_finally << T_EOL
      << NonTerminal::endr;

   S_Raise << "Stmt-raise" << raise_errhand
      << NonTerminal::sr
      << "r_raise" << apply_raise << T_raise << Expr << T_EOL
      << NonTerminal::endr;

   CatchSpec << "CatchSpec"
      << NonTerminal::sr
      << "all" << apply_catch_all << T_EOL
      << "in-var" << apply_catch_in_var << T_in << T_Name << T_EOL
      << "as-var" << apply_catch_as_var << T_as << T_Name << T_EOL
      << "thing" << apply_catch_thing <<  CaseList << T_EOL
      << "thing-in-var" << apply_catch_thing_in_var << CaseList << T_in << T_Name << T_EOL
      << "thing-as-var" << apply_catch_thing_as_var << CaseList << T_as << T_Name << T_EOL
      << NonTerminal::endr;

//========================================================================================
// Load, import & family
//

   S_Load << "Stmt-load" << load_errhand
       << NonTerminal::sr
       << "load-string" << apply_load_string << T_load << T_String << T_EOL
       << "load-modspec" << apply_load_mod_spec << T_load << ModSpec << T_EOL
       << NonTerminal::endr;

   // We use a selector strategy to reduce the amount of tokens visible to the root state.s
   S_Import << "Stmt-import" << import_errhand
      << NonTerminal::sr
      << apply_import << T_import << ImportClause << T_EOL
      << NonTerminal::endr;


   S_Export << "Stmt-export" << export_errhand
      << NonTerminal::sr
      << apply_export_rule << T_export << ListSymbol << T_EOL
      << NonTerminal::endr;


   S_Namespace << "Stmt-namespace" << namespace_errhand
      << NonTerminal::sr
      << apply_namespace << T_namespace << NameSpaceSpec << T_EOL
      << NonTerminal::endr;


   ModSpec << "ModSpec" << load_modspec_errhand
      << NonTerminal::sr
      << "next" << apply_modspec_next << ModSpec << T_Dot << T_Name
      << "first_dot" << apply_modspec_first_dot << T_Dot << T_Name
      << "first_self" << apply_modspec_first_self << T_self
      << "first" << apply_modspec_first << T_Name
      << NonTerminal::endr;


   ImportClause << "ImportClause"
        << NonTerminal::sr
        << "star_from_string_in" << apply_import_star_from_string_in << T_Times << T_from << T_String << T_in << NameSpaceSpec << T_EOL
        << "star_from_string" << apply_import_star_from_string << T_Times << T_from << T_String << T_EOL
        << "star_from_modspec_in" << apply_import_star_from_modspec_in << T_Times << T_from << ModSpec << T_in << NameSpaceSpec << T_EOL
        << "star_from_modspec_in" << apply_import_star_from_modspec << T_Times << T_from << ModSpec << T_EOL
        << "from_string_as" << apply_import_from_string_as << ImportSpec << T_from << T_String << T_as << T_Name << T_EOL
        << "from_string_in" << apply_import_from_string_in << ImportSpec << T_from << T_String << T_in << NameSpaceSpec << T_EOL
        << "from_string" << apply_import_string << ImportSpec << T_from << T_String << T_EOL
        << "from_modspec_as" << apply_import_from_modspec_as << ImportSpec << T_from << ModSpec << T_as << T_Name << T_EOL
        << "from_modspec_in" << apply_import_from_modspec_in << ImportSpec << T_from << ModSpec << T_in << NameSpaceSpec << T_EOL
        << "from_modspec" << apply_import_from_modspec << ImportSpec << T_from << ModSpec << T_EOL
        << "syms" << apply_import_syms << ImportSpec << T_EOL
        << NonTerminal::endr;


   ImportSpec << "ImportSpec" << importspec_errhand
            << NonTerminal::sr
            << "next" << apply_ImportSpec_next << ImportSpec << T_Comma << NameSpaceSpec
            << "attach_last" << apply_ImportSpec_attach_last << ImportSpec << T_Dot << T_Times
            << "attach_next" << apply_ImportSpec_attach_next << ImportSpec << T_Dot << T_Name
            << "first" << apply_ImportSpec_first << NameSpaceSpec
            << "empty" << apply_ImportSpec_empty
            << NonTerminal::endr;


   NameSpaceSpec << "NameSpaceSpec"
       << NonTerminal::sr
       << "last" << apply_nsspec_last << NameSpaceSpec << T_Dot << T_Times
       << "next" << apply_nsspec_next << NameSpaceSpec << T_Dot << T_Name
       << "first" << apply_nsspec_first << T_Name
       << NonTerminal::endr;

//========================================================================================
// Top level structures
//

   S_Function << "Stmt-function"
      << NonTerminal::sr
      << "decl" << apply_function << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL
      << "ETA decl" << apply_function_eta << T_function << T_Amper << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL
      << NonTerminal::endr;

   S_ProtoProp << "Stmt-protoprop"
         << NonTerminal::sr
         << "proto_prop" << apply_proto_prop  << T_Name << T_EqSign << Expr << T_EOL
         << NonTerminal::endr;

   S_Class << "Stmt-class" << classdecl_errhand
         << NonTerminal::sr
         << "Class decl" << apply_class_statement << T_class << T_Name
         << NonTerminal::endr;

   S_Object << "Stmt-object" << classdecl_errhand
         << NonTerminal::sr
         << "Object decl" << apply_object_statement << T_object << T_Name
         << NonTerminal::endr;

   S_PropDecl << "Stmt-porpdecl"
         << NonTerminal::sr
         << "Expression Property" << apply_pdecl_expr << T_Name << T_EqSign << Expr << T_EOL
         << NonTerminal::endr;


   S_StaticPropDecl << "Stmt-static-pdecl"
         << NonTerminal::sr
         << "pdecl" << apply_static_pdecl_expr << T_static << T_Name << T_EqSign << Expr << T_EOL
         << "fdecl" << apply_static_function << T_static << T_function << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL
         << "ETA decl" << apply_static_function_eta << T_static << T_function << T_Amper << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL
         << NonTerminal::endr;

   S_InitDecl << "Init block"
         << NonTerminal::sr
         << apply_init_expr << T_init << T_EOL
         << NonTerminal::endr;

   //==========================================================================
   // Toplevel declaration components
   //

   FromClause << "From-clause"
      << NonTerminal::sr
      << "next" << apply_FromClause_next << FromClause << T_Comma << FromEntry
      << "FromClause_first" << apply_FromClause_first << FromEntry
      << NonTerminal::endr;

   FromEntry << "From-entry"
      << NonTerminal::sr
      << "w/expr" << apply_FromClause_entry_with_expr << T_Name << T_Openpar << ListExpr << T_Closepar
      << "w/o expr" << apply_FromClause_entry << T_Name
      << NonTerminal::endr;

   ClassParams << "ClassParams" << classdecl_errhand
      << NonTerminal::sr
      << "w/from" << apply_class_from << T_from << FromClause << T_EOL
      << "simple" << apply_class << T_EOL
      << "w/params & from" << apply_class_p_from << T_Openpar << ListSymbol << T_Closepar << T_from << FromClause << T_EOL
      << "w/params" << apply_class_p  << T_Openpar << ListSymbol << T_Closepar  << T_EOL
      << NonTerminal::endr;

   // Objects are like classes, but they don't use parameters.
   ObjectParams << "ObjectParams" << classdecl_errhand
      << NonTerminal::sr
      << "w/from" << apply_class_from << T_from << FromClause << T_EOL
      << "simple" << apply_class << T_EOL
      << NonTerminal::endr;

//========================================================================================
// Main
//
   /*
   StdOutStream sout;
   TextWriter tw(&sout);
   MainProgram.render(tw);
   */

   addState( MainProgram );
   addState( ClassBody );
   addState( InlineFunc );
   addState( LambdaStart );
   addState( EPState );
   addState( ClassStart );
   addState( ObjectStart );
   addState( ProtoDecl );
   addState( ArrayDecl );
}

LSourceParser::~LSourceParser()
{
   TRACE("Destroying SourceParser 0x%p",this);
}

void LSourceParser::onPushState( bool isPushedState )
{
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   pc->onStatePushed( isPushedState );
}


void LSourceParser::onPopState()
{
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   pc->onStatePopped();
}

bool LSourceParser::parse()
{
   // we need a context (and better to be a SourceContext
   if ( m_ctx == 0 )
   {
      throw new CodeError( ErrorParam( e_setup, __LINE__, SRC ).extra("LSourceParser::parse - setContext") );
   }

   bool result =  Parser::parse("Main");
   if( result ) {
      ParserContext* pc = static_cast<ParserContext*>(m_ctx);
      pc->onInputOver();
      result = ! hasErrors();
   }
   return result;
}

void LSourceParser::reset()
{
   Parser::reset();
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
   pc->reset();
}


void LSourceParser::addError( int code, const String& uri, int l, int c, int ctx, const String& extra )
{
#ifndef NDEBUG
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
#endif
   Parser::addError( code, uri, l, c, ctx, extra );
}

void LSourceParser::addError( Error* err )
{
#ifndef NDEBUG
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
#endif
   Parser::addError( err );
}


void LSourceParser::addError( int code, const String& uri, int l, int c, int ctx )
{
#ifndef NDEBUG
   ParserContext* pc = static_cast<ParserContext*>(m_ctx);
   fassert( pc != 0 );
#endif
   Parser::addError( code, uri, l, c, ctx );
}




//=========================================================================================
//=========================================================================================
//=========================================================================================
//=========================================================================================
//=========================================================================================





class FALCON_DYN_CLASS DynCompilerCtx: public ParserContext
{
public:
   DynCompilerCtx(VMContext* ctx, SourceParser& sp):
      ParserContext( &sp ),
      m_ctx(ctx),
      m_sp(sp)
   {}

   virtual ~DynCompilerCtx(){}

   virtual void onInputOver() {}

   virtual bool onOpenFunc( Function* func ) {
      FALCON_GC_HANDLE( func );
      return true;
   }

   virtual void onCloseFunc( Function* f) {
      if( f->name() == "") f->name("$anon");
   }

   virtual bool onOpenClass( Class* cls, bool isObj ) {
      FALCON_GC_HANDLE( cls );

      if( isObj ) {
         m_sp.addError( e_toplevel_obj, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
         return false;
      }
      return true;
   }

   virtual void onOpenMethod( Class* cls, Function* func) {
      static Variable var(Variable::e_nt_undefined, Variable::undef, 0, true);
      if( static_cast<FalconClass*>(cls)->getProperty(func->name())) {
         m_sp.addError( e_prop_adef, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
      }
   }


   virtual void onCloseClass( Class* f, bool ) {
      if( f->name() == "") f->name("$anon");
   }

   virtual void onNewStatement( TreeStep* ) {}

   virtual void onLoad( const String&, bool ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }

   virtual bool onImportFrom( ImportDef*  ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
      return false;
   }

   virtual void onExport(const String& ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }
   virtual void onDirective(const String& , const String& ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }
   virtual void onGlobal( const String&  ) {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }

   virtual void onGlobalDefined( const String& , bool&  ) {
   }

   virtual bool onGlobalAccessed( const String& ) {
      return false;
   }

   virtual Item* getValue( Symbol* sym ) {
      Item* value = m_ctx->resolveSymbol( sym, false);
      return value;
   }

   virtual bool onAttribute(const String& name, TreeStep* generator, Mantra* target )
   {
      SourceParser& sp = m_sp;
      if( target == 0 )
      {
         sp.addError( e_directive_not_allowed, sp.currentSource(), sp.currentLine()-1, 0, 0 );
      }
      else
      {
         return attribute_helper(m_ctx, name, generator, target );
      }

      return true;
   }

   virtual void onRequirement( Requirement*  )
   {
      m_sp.addError( e_directive_not_allowed, m_sp.currentSource(), m_sp.currentLine()-1, 0, 0 );
   }

   virtual void onIString( const String& )
   {
      // do nothing;
   }

private:
   VMContext* m_ctx;
   SourceParser& m_sp;
};

SynTree* DynCompiler::compile( const String& str, SynTree* target )
{
   static Engine* eng = Engine::instance();

   StringStream ss(str);
   Transcoder* tc;
   switch( str.manipulator()->charSize() )
   {
   case 2: tc = eng->getTranscoder("F16"); break;
   case 4: tc = eng->getTranscoder("F32"); break;
   default: tc = eng->getTranscoder("C"); break;
   }

   return compile(&ss, tc, target);
}

SynTree* DynCompiler::compile( Stream* stream, Transcoder* tc, SynTree* target )
{
   if( tc == 0 )
   {
      LocalRef<TextReader>tr( new TextReader(stream, tc ));
      return compile(&tr, target );
   }
   else {
      LocalRef<TextReader>tr(new TextReader (stream ));
      return compile(&tr, target );
   }
}


SynTree* DynCompiler::compile( TextReader* reader, SynTree* target)
{
   // check the parameters.
   // prepare the parser
   SourceParser sp;
   DynCompilerCtx compctx( m_ctx, sp );
   sp.setContext( &compctx );

   // start parsing.
   SourceLexer* slex = new SourceLexer( "<internal>", &sp, reader );
   SynTree* st = target == 0 ? new SynTree : target;
   sp.pushLexer(slex);

   try
   {
      compctx.openMain( st );

      //TODO: idle the context
      if( ! sp.parse() ) {
         // todo: eventually re-box the error?
         throw sp.makeError();
      }

      return st;
   }
   catch( Error* e )
   {
      if( target == 0 )
      {
         delete st;
      }

      throw e;
   }
   return 0;
}

}

/* end of dyncompiler.cpp */
