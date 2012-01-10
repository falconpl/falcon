/*
   FALCON - The Falcon Programming Language.
   FILE: synclasses.cpp

   Class holding all the Class reflector for syntactic tree elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 29 Dec 2011 12:07:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/synclasses.cpp"

#include <falcon/setup.h>
#include <falcon/synclasses.h>
#include <falcon/vmcontext.h>
#include <falcon/engine.h>
#include <falcon/treestep.h>

#include <falcon/synclasses_id.h>

#include <falcon/errors/paramerror.h>

#include <falcon/psteps/breakpoint.h>
#include <falcon/psteps/exprarray.h>
#include <falcon/psteps/exprassign.h>
#include <falcon/psteps/exprbitwise.h>
#include <falcon/psteps/exprcall.h>
#include <falcon/psteps/exprclosure.h>
#include <falcon/psteps/exprcompare.h>
#include <falcon/psteps/exprdict.h>
#include <falcon/psteps/exprdot.h>
#include <falcon/psteps/expreeq.h>
#include <falcon/psteps/expreval.h>
#include <falcon/psteps/expriif.h>
#include <falcon/psteps/exprincdec.h>
#include <falcon/psteps/exprindex.h>
#include <falcon/psteps/exprlit.h>
#include <falcon/psteps/exprlogic.h>
#include <falcon/psteps/exprmath.h>
#include <falcon/psteps/exprmultiunpack.h>
#include <falcon/psteps/exprneg.h>
#include <falcon/psteps/exproob.h>
#include <falcon/psteps/exprproto.h>
#include <falcon/psteps/exprrange.h>
#include <falcon/psteps/exprref.h>
#include <falcon/psteps/exprself.h>
#include <falcon/psteps/exprstarindex.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprunpack.h>
#include <falcon/psteps/exprvalue.h>

#include <falcon/psteps/stmtautoexpr.h>
#include <falcon/psteps/stmtbreak.h>
#include <falcon/psteps/stmtcontinue.h>
#include <falcon/psteps/stmtfastprint.h>
#include <falcon/psteps/stmtfor.h>
#include <falcon/psteps/stmtif.h>
#include <falcon/psteps/stmtinit.h>
#include <falcon/psteps/stmtraise.h>
#include <falcon/psteps/stmtreturn.h>
#include <falcon/psteps/stmtrule.h>
#include <falcon/psteps/stmtselect.h>
#include <falcon/psteps/stmttry.h>
#include <falcon/psteps/stmtwhile.h>

#include <falcon/itemarray.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>

namespace Falcon {
#undef FALCON_SYNCLASS_DECLARATOR_DECLARE
#define FALCON_SYNCLASS_DECLARATOR_APPLY

SynClasses::SynClasses( Class* classSynTree, Class* classStatement, Class* classExpr ):
   m_cls_st( classSynTree ),
   m_cls_stmt( classStatement ),
   m_cls_expr( classExpr ),
   #include <falcon/synclasses_list.h>
   m_dummy_end(0)
{
   m_stmt_forto.userFlags(FALCON_SYNCLASS_ID_FORCLASSES);
   m_stmt_forin.userFlags(FALCON_SYNCLASS_ID_RULE);
   m_stmt_if.userFlags(FALCON_SYNCLASS_ID_ELSEHOST);
   m_stmt_select.userFlags(FALCON_SYNCLASS_ID_CASEHOST);
   m_stmt_autoexpr.userFlags(FALCON_SYNCLASS_ID_AUTOEXPR);
   m_expr_call.userFlags(FALCON_SYNCLASS_ID_CALLFUNC);
}

SynClasses::~SynClasses() {}
 
void SynClasses::subscribe( Engine* engine )
{
   #undef FALCON_SYNCLASS_DECLARATOR_APPLY
   #define FALCON_SYNCLASS_DECLARATOR_REGISTER
   #include <falcon/synclasses_list.h>
}
   

void SynClasses::varExprInsert( VMContext* ctx, int pcount, TreeStep* step )
{
   Item* operands = ctx->opcodeParams(pcount);
   int count = 0;
   while( count < pcount )
   {
      bool bCreate = true;
      TreeStep* ts = TreeStep::checkExpr(operands[count++], bCreate);
      if( ts == 0 )
      {
         delete step;
         throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Incompatible entity at ").N(count) ) );
      }
      
      // larger than size, as we ++count before, but it's ok.
      if( ! step->insert(count, ts) )
      {
         delete step;
         // theoretically parented entities are not created, but...
         if ( bCreate ) delete ts;
         
         // params count 1 to N so we're ok to use count that has been ++
         throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Already parented entity at ").N(count) ) );
      }
   }
}
   

void SynClasses::naryExprSet( VMContext* ctx, int pcount, TreeStep* step, int32 size )
{
   if( pcount < size )
   {
      delete step;
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Expression|Symbol|X") ) );
   }
   
   Item* operands = ctx->opcodeParams(pcount);
   int count = 0;
   while( count < size )
   {
      bool bCreate = true;
      TreeStep* ts = TreeStep::checkExpr(operands[count], bCreate);
      if( ts == 0 )
      {
         delete step;
         throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Incompatible entity at ").N(count+1) ) );
      }
      
      // Set the nth element.
      if( ! step->nth(count, ts) )
      {
         delete step;
         if ( bCreate ) delete ts;
         
         throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Already parented entity at ").N(count+1) ) );
      }
      ++count;
   }
}

GCToken* SynClasses::collect( const Class* cls, TreeStep* earr, int line )
{
   FALCON_UNUSED_PARAM(line)
   static Collector* coll = Engine::instance()->collector();
   return FALCON_GC_STORE_PARAMS( coll, cls, earr, line, SRC );
}

//===========================================================
// The clases
//

#define FALCON_STANDARD_SYNCLASS_OP_CREATE( cls, exprcls, operation ) \
   void SynClasses::Class## cls ::op_create( VMContext* ctx, int pcount ) const\
   {\
      exprcls * expr = new exprcls ; \
      SynClasses:: operation ( ctx, pcount, expr ); \
      ctx->stackResult( pcount+1, SynClasses::collect( this, expr, __LINE__ ) ); \
   }\
   void SynClasses::Class##cls ::restore( VMContext* ctx, DataReader*dr, void*& empty ) const \
   {\
      empty = new exprcls ; \
      m_parent->restore( ctx, dr, empty ); \
   }\


FALCON_STANDARD_SYNCLASS_OP_CREATE( GenArray, ExprArray, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Assign, ExprAssign, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( BNot, ExprBNOT, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Call, ExprCall, varExprInsert )
// GenClosure --specificly managed

FALCON_STANDARD_SYNCLASS_OP_CREATE( LT, ExprLT, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( LE, ExprLE, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( GT, ExprGT, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( GE, ExprGE, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( EQ, ExprEQ, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( NE, ExprNE, binaryExprSet )
   
// GenDict --specificly managed
// DotAccess -- specificly managed
FALCON_STANDARD_SYNCLASS_OP_CREATE( EEQ, ExprEEQ, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Eval, ExprEval, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( IIF, ExprIIF, ternaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Lit, ExprLit, unaryExprSet )

// inc-dec
FALCON_STANDARD_SYNCLASS_OP_CREATE( PreInc, ExprPreInc, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( PreDec, ExprPreDec, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( PostInc, ExprPostInc, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( PostDec, ExprPostDec, unaryExprSet )
   
FALCON_STANDARD_SYNCLASS_OP_CREATE( IndexAccess, ExprIndex, binaryExprSet )

// Logic
FALCON_STANDARD_SYNCLASS_OP_CREATE( Not, ExprNot, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( And, ExprAnd, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Or, ExprOr, binaryExprSet )

//Math
FALCON_STANDARD_SYNCLASS_OP_CREATE( Plus, ExprPlus, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Minus, ExprMinus, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Times, ExprTimes, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Div, ExprDiv, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Mod, ExprMod, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Pow, ExprPow, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( LShift, ExprLShift, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( RShift, ExprRShift, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( BAnd, ExprBAND, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( BOr, ExprBOR, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( BXor, ExprBXOR, binaryExprSet )
//Auto-Math
FALCON_STANDARD_SYNCLASS_OP_CREATE( AutoPlus, ExprAutoPlus, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( AutoMinus, ExprAutoMinus, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( AutoTimes, ExprAutoTimes, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( AutoDiv, ExprAutoDiv, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( AutoMod, ExprAutoMod, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( AutoPow, ExprAutoPow, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( AutoLShift, ExprAutoLShift, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( AutoRShift, ExprAutoRShift, binaryExprSet )

// MUnpack -- separated
FALCON_STANDARD_SYNCLASS_OP_CREATE( Neg, ExprNeg, unaryExprSet )

// OOB
FALCON_STANDARD_SYNCLASS_OP_CREATE( Oob, ExprOob, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( DeOob, ExprDeoob, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( IsOob, ExprIsOob, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( XorOob, ExprXorOob, unaryExprSet )

// GenProto -- separated
// GenRange -- seperated
// GenRef -- separated
FALCON_STANDARD_SYNCLASS_OP_CREATE( Self, ExprSelf, zeroaryExprSet )

FALCON_STANDARD_SYNCLASS_OP_CREATE( StarIndexAccess, ExprStarIndex, binaryExprSet )

// Sym -- separated
// Unpack -- separated
// Value -- separated
   
//=================================================================
// Specific management
//

void SynClasses::ClassGenClosure::op_create( VMContext* ctx, int pcount ) const
{
   if( pcount < 1 || ! ctx->topData().isFunction() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("C") ) );
   }
   
   ExprClosure* expr = new ExprClosure( ctx->topData().asFunction() );
   ctx->stackResult( pcount+1, SynClasses::collect( this, expr, __LINE__ ) );
}
void SynClasses::ClassGenClosure::restore( VMContext* ctx, DataReader*dr, void*& empty ) const
{
   // TODO
   empty = new ExprClosure;
   m_parent->restore( ctx, dr, empty );
}


void SynClasses::ClassGenDict::op_create( VMContext* ctx, int pcount ) const
{
   if( pcount % 2 != 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Need pair count of expressions") ) );
   }
   
   ExprDict* expr = new ExprDict;
   SynClasses:: varExprInsert( ctx, pcount, expr );
   ctx->stackResult( pcount+1, SynClasses::collect( this, expr, __LINE__ ) );
}
void SynClasses::ClassGenDict::restore( VMContext* ctx, DataReader*dr, void*& empty ) const
{
   empty = new ExprDict;
   m_parent->restore( ctx, dr, empty );
}



void SynClasses::ClassDotAccess::op_create( VMContext* ctx, int pcount ) const
{
   if( (pcount < 2) || (! ctx->opcodeParams(pcount)[1].isString()) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Expression,S") ) );
   }
   
   ExprDot* expr = new ExprDot;
   SynClasses::unaryExprSet( ctx, pcount, expr );
   expr->property( *ctx->opcodeParams(pcount)[1].asString() );
   ctx->stackResult( pcount+1, SynClasses::collect( this, expr, __LINE__ ) );
}
void SynClasses::ClassDotAccess::restore( VMContext* ctx, DataReader*dr, void*& empty ) const
{
   empty = new ExprDot;
   m_parent->restore( ctx, dr, empty );
}



void SynClasses::ClassMUnpack::op_create( VMContext* ctx, int pcount ) const
{       
   // TODO -- parse a list of pairs symbol->expression
   Class::op_create( ctx, pcount );
}
void SynClasses::ClassMUnpack::restore( VMContext* ctx, DataReader*dr, void*& empty ) const
{
   empty = new ExprMultiUnpack;
   m_parent->restore( ctx, dr, empty );
}


void SynClasses::ClassGenProto::op_create( VMContext* ctx, int pcount ) const
{       
   // TODO -- parse a list of pairs string->expression
   Class::op_create( ctx, pcount );
}
void SynClasses::ClassGenProto::restore( VMContext* ctx, DataReader*dr, void*& empty ) const
{
   empty = new ExprProto;
   m_parent->restore( ctx, dr, empty );
}

void SynClasses::ClassUnpack::op_create( VMContext* ctx, int pcount ) const
{       
   // TODO -- parse a list of pairs symbol, + 1 terminal expression
   Class::op_create( ctx, pcount );
}
void SynClasses::ClassUnpack::restore( VMContext* ctx, DataReader*dr, void*& empty ) const
{
   empty = new ExprUnpack;
   m_parent->restore( ctx, dr, empty );
}


void SynClasses::ClassGenRange::op_create( VMContext* ctx, int pcount ) const
{
   if( pcount < 3 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Expression|Nil,Expression|Nil,Expression|Nil") ) );
   }
      
   ExprRange* rng = new ExprRange;
   Item* ops = ctx->opcodeParams(pcount);
   
   bool bOk = true;
   if( ! ops[0].isNil() )
   {
      bool bCreate = true;
      Expression* val = static_cast<Expression*>(TreeStep::checkExpr(ops[0],bCreate));
      // TODO: Check for integers.
      if( val == 0 ) bOk = false;
      rng->start( val );
   }
   
   if( bOk && ! ops[1].isNil() )
   {
      bool bCreate = true;
      Expression* val = static_cast<Expression*>(TreeStep::checkExpr(ops[1],bCreate));
      // TODO: Check for integers.
      if( val == 0 ) bOk = false;
      rng->end( val );
   }

   if( bOk && ! ops[2].isNil() )
   {
      bool bCreate = true;
      Expression* val = static_cast<Expression*>(TreeStep::checkExpr(ops[2],bCreate));
      // TODO: Check for integers.
      if( val == 0 ) bOk = false;      
      rng->step( val );
   }
   
   if( ! bOk )
   {
      delete rng;
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_runtime)
         .extra( String("Expression|Nil,Expression|Nil,Expression|Nil") ) );
   }

   ctx->stackResult( pcount+1, SynClasses::collect( this, rng, __LINE__ ) );
}

void SynClasses::ClassGenRange::restore( VMContext* ctx, DataReader*dr, void*& empty ) const
{
   empty = new ExprRange;
   m_parent->restore( ctx, dr, empty );
}

void SynClasses::ClassGenRef::op_create( VMContext* ctx, int pcount ) const
{
   static Class* symClass = Engine::instance()->symbolClass();
   static Class* exprClass = Engine::instance()->expressionClass();
   
   Item* params = ctx->opcodeParams(pcount);
   Class* cls=0;
   void* data=0;
   if( pcount < 1 || params->asClassInst(cls, data) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Symbol|GenSym") ) );
   }
   
   if( cls->isDerivedFrom( symClass ) )
   {
      //TODO:TreeStepInherit
      Symbol* sym = static_cast<Symbol*>( data );
      ExprRef* expr = new ExprRef( sym );
      ctx->stackResult( pcount+1, SynClasses::collect( this, expr, __LINE__ ) );
   }
   else if( cls->isDerivedFrom( exprClass ) &&
      static_cast<Expression*>(data)->trait() == Expression::e_trait_symbol )
   {
      //TODO:TreeStepInherit
      ExprSymbol* sym = static_cast<ExprSymbol*>( data );
      ExprRef* expr = new ExprRef( sym );
      ctx->stackResult( pcount+1, SynClasses::collect( this, expr, __LINE__ ) );
   }
   else {            
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Symbol|Sym") ) );
   }
}
void SynClasses::ClassGenRef::restore( VMContext* ctx, DataReader*dr, void*& empty ) const
{
   empty = new ExprRef;
   m_parent->restore( ctx, dr, empty );
}


void SynClasses::ClassGenSym::op_create( VMContext* ctx, int pcount ) const
{
   static Class* symClass = Engine::instance()->symbolClass();
   
   Item* params = ctx->opcodeParams(pcount);
   
   if( pcount >= 1 )
   {
      Class* cls;
      void* data;
      if( params->isString() )
      {
         ExprSymbol* expr = new ExprSymbol( *params->asString() );
         ctx->stackResult( pcount+1, SynClasses::collect( this, expr, __LINE__ ) );
         return;
      }
      else if( params->asClassInst(cls, data) && cls->isDerivedFrom(symClass) )
      {
         Symbol* sym = static_cast<Symbol*>( data );
         ExprSymbol* expr = new ExprSymbol( sym );
         ctx->stackResult( pcount+1, SynClasses::collect( this, expr, __LINE__ ) );
         return;
      }
   }
      
   throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_runtime)
         .extra( String("Symbol|S") ) );
   //TODO:TreeStepInherit
}

void SynClasses::ClassGenSym::store( VMContext*, DataWriter* dw, void* instance ) const
{
   ExprSymbol* es = static_cast<ExprSymbol*>(instance);
   dw->write( es->line() );
   dw->write( es->chr() );
   dw->write( es->name() );
}

void SynClasses::ClassGenSym::restore( VMContext*, DataReader*dr, void*& empty ) const
{
   ExprSymbol* es = new ExprSymbol;   
   int32 line, chr;
   String name;
   
   // TODO: this is just a test.
   dr->read( line );
   dr->read( chr );
   dr->read( name );
   es->decl( line, chr );
   es->name( name );
   empty = es;
}


void SynClasses::ClassValue::op_create( VMContext* ctx, int pcount ) const
{
   if( pcount < 1 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("X") ) );
   }
   
   ExprValue* expr = new ExprValue( *ctx->opcodeParams(pcount) );
   ctx->stackResult( pcount+1, SynClasses::collect( this, expr, __LINE__ ) );
}

void SynClasses::ClassValue::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ExprValue* ev = static_cast<ExprValue*>( instance );
   subItems.resize(1);
   subItems[0] = ev->item();
}

void SynClasses::ClassValue::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   fassert(subItems.length() == 1);
   ExprValue* ev = static_cast<ExprValue*>( instance );
   ev->item( subItems[0] );
}

void SynClasses::ClassValue::restore( VMContext* ctx, DataReader*dr, void*& empty ) const
{
   empty = new ExprValue;
   m_parent->restore( ctx, dr, empty );
}

//=================================================================
// Statements
//

//TODO: Proper constructor -- for now just ignore constructors.
FALCON_STANDARD_SYNCLASS_OP_CREATE( AutoExpr, StmtAutoexpr, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Break, StmtBreak, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Breakpoint, Breakpoint, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Continue, StmtContinue, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Cut, StmtCut, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Doubt, StmtDoubt, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( FastPrint, StmtFastPrint, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( ForIn, StmtForIn, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( ForTo, StmtForTo, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( If, StmtIf, zeroaryExprSet )
   
// Init should be reviewed as we introduce the class declaration expression.
FALCON_STANDARD_SYNCLASS_OP_CREATE( Init, StmtInit, zeroaryExprSet )
   
FALCON_STANDARD_SYNCLASS_OP_CREATE( Raise, StmtRaise, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Return, StmtReturn, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Rule, StmtRule, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Select, StmtSelect, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Try, StmtTry, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( While, StmtWhile, zeroaryExprSet )


//=================================================================
// Trees
//
FALCON_STANDARD_SYNCLASS_OP_CREATE( RuleSynTree, RuleSynTree, zeroaryExprSet )
   
}

/* end of synclasses.cpp */
