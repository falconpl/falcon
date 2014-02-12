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
   virtual ~SourceParser();
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

}

#endif	/* SOURCEPARSER_H */

/* end of sourceparser.h */
