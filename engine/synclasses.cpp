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

#include <falcon/synclasses.h>
#include <falcon/vmcontext.h>
#include <falcon/engine.h>
   

namespace Falcon {

SynClasses::~SynClasses() {}
 
void SynClasses::subscribe( Engine* engine )
{
   engine->addBuiltin( & m_expr_genarray);
   engine->addBuiltin( & m_expr_assign);
   engine->addBuiltin( & m_expr_bnot);
   engine->addBuiltin( & m_expr_call);
   // compare
   engine->addBuiltin( & m_expr_lt);
   engine->addBuiltin( & m_expr_le);
   engine->addBuiltin( & m_expr_gt);
   engine->addBuiltin( & m_expr_ge);
   engine->addBuiltin( & m_expr_eq);
   engine->addBuiltin( & m_expr_ne);
   //
   engine->addBuiltin( & m_expr_gendict);
   engine->addBuiltin( & m_expr_dot);
   engine->addBuiltin( & m_expr_eeq);
   engine->addBuiltin( & m_expr_iif);
   // inc-dec
   engine->addBuiltin( & m_expr_preinc);
   engine->addBuiltin( & m_expr_predec);
   engine->addBuiltin( & m_expr_postinc);
   engine->addBuiltin( & m_expr_postdec);
   //
   engine->addBuiltin( & m_expr_index);
   // Logic
   engine->addBuiltin( & m_expr_not);
   engine->addBuiltin( & m_expr_and);
   engine->addBuiltin( & m_expr_or);
   // Math
   engine->addBuiltin( & m_expr_plus);
   engine->addBuiltin( & m_expr_minus);
   engine->addBuiltin( & m_expr_times);
   engine->addBuiltin( & m_expr_div);
   engine->addBuiltin( & m_expr_mod);
   engine->addBuiltin( & m_expr_pow);
   // Auto-math
   engine->addBuiltin( & m_expr_aplus);
   engine->addBuiltin( & m_expr_aminus);
   engine->addBuiltin( & m_expr_atimes);
   engine->addBuiltin( & m_expr_adiv);
   engine->addBuiltin( & m_expr_amod);
   engine->addBuiltin( & m_expr_apow);
   // 
   engine->addBuiltin( & m_expr_munpack);
   engine->addBuiltin( & m_expr_neg);
   // OOB
   engine->addBuiltin( & m_expr_oob);
   engine->addBuiltin( & m_expr_deoob);
   engine->addBuiltin( & m_expr_isoob);
   engine->addBuiltin( & m_expr_xoroob);
   // 
   engine->addBuiltin( & m_expr_genproto);
   engine->addBuiltin( & m_expr_genrange);
   engine->addBuiltin( & m_expr_genref);
   engine->addBuiltin( & m_expr_self);
   engine->addBuiltin( & m_expr_starindex);
   engine->addBuiltin( & m_expr_sym);
   engine->addBuiltin( & m_expr_unpack);
   engine->addBuiltin( & m_expr_value);
   
   //======================================================================
   // Statement classes
   //
   engine->addBuiltin( & m_stmt_autoexpr);
   engine->addBuiltin( & m_stmt_break);
   engine->addBuiltin( & m_stmt_breakpoint);
   engine->addBuiltin( & m_stmt_continue);
   engine->addBuiltin( & m_stmt_fastprint);
   engine->addBuiltin( & m_stmt_forin);
   engine->addBuiltin( & m_stmt_forto);
   engine->addBuiltin( & m_stmt_if);
   engine->addBuiltin( & m_stmt_init);
   engine->addBuiltin( & m_stmt_raise);
   engine->addBuiltin( & m_stmt_return);
   engine->addBuiltin( & m_stmt_rule);
   engine->addBuiltin( & m_stmt_select);
   engine->addBuiltin( & m_stmt_try);
   engine->addBuiltin( & m_stmt_while);
   
   //======================================================================
   // Syntree classes
   //
   engine->addBuiltin( & m_st_rulest);
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

GarbageToken* SynClasses::collect( Class*, TreeStep*, int line )
{
   const Collector* coll = Engine::instance()->collector();
   return FALCON_GC_STORE_PARAMS( coll, this, earr, line, SRC );
}

//===========================================================
// The clases
//

#define FALCON_STANDARD_SYNCLASS_OP_CREATE( cls, exprcls, operation ) \
   void SynClasses::Class## cls ::op_create( VMContext* ctx, int pcount ) const\
   {\
      ##exprcls * expr = new ##exprcls ; \
      SynClasses:: ##operation ( ctx, pcount, expr ); \
      ctx->stackResult( pcount, SynClasses::collect( this, expr, __LINE__ ) ); \
   }\
   TreeStep* SynClasses::Class## cls ::createInstance() const\
   {\
      return new ##exprcls ; \
   }


FALCON_STANDARD_SYNCLASS_OP_CREATE( GenArray, ExprArray, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Assign, ExprAssign, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( BNot, ExprBNot, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Call, ExprCall, varExprInsert )

FALCON_STANDARD_SYNCLASS_OP_CREATE( LT, ExprLT, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( LE, ExprLE, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( GT, ExprGT, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( GE, ExprGE, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( EQ, ExprEQ, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( NE, ExprNE, binaryExprSet )
   
// GenDict --specificly managed
// DotAccess -- specificly managed
FALCON_STANDARD_SYNCLASS_OP_CREATE( EEQ, ExprEEQ, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( IIF, ExprIIF, ternaryExprSet )

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
   ctx->stackResult( pcount, SynClasses::collect( this, expr, __LINE__ ) );
}
TreeStep* SynClasses::ClassGenDict::createInstance() const { return new ExprDict; }



void SynClasses::ClassDotAccess::op_create( VMContext* ctx, int pcount ) const
{
   if( pcount < 2 != 0 || ! ctx->opcodeParams(pcount)[1].isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Expression,S") ) );
   }
   
   ExprDot* expr = new ExprDot;
   SynClasses::unaryExprSet( ctx, pcount, expr );
   expr->property( *ctx->opcodeParams(pcount)[1].asString() );
   ctx->stackResult( pcount, SynClasses::collect( this, expr, __LINE__ ) );
}
TreeStep* SynClasses::ClassDotAccess::createInstance() const { return new ExprDot; }


void SynClasses::ClassMUnpack::op_create( VMContext* ctx, int pcount ) const
{       
   // TODO -- parse a list of pairs symbol->expression
   Class::op_create( ctx, pcount );
}
TreeStep* SynClasses::ClassMUnpack::createInstance() const { return new ExprMultiUnpack; }


void SynClasses::ClassGenProto::op_create( VMContext* ctx, int pcount ) const
{       
   // TODO -- parse a list of pairs string->expression
   Class::op_create( ctx, pcount );
}
TreeStep* SynClasses::ClassGenProto::createInstance() const { return new ExprProto; }

void SynClasses::ClassUnpack::op_create( VMContext* ctx, int pcount ) const
{       
   // TODO -- parse a list of pairs symbol, + 1 terminal expression
   Class::op_create( ctx, pcount );
}
TreeStep* SynClasses::ClassUnpack::createInstance() const { return new ExprUnpack; }


void SynClasses::ClassGenRange::op_create( VMContext* ctx, int pcount ) const
{
   if( pcount < 3 != 0 )
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
      Expression* val = TreeStep::checkExpr(ops[0],bCreate);
      // TODO: Check for integers.
      if( val == 0 ) bOk = false;
      rng->start( val );
   }
   
   if( bOk && ! ops[1].isNil() )
   {
      bool bCreate = true;
      Expression* val = TreeStep::checkExpr(ops[1],bCreate);
      // TODO: Check for integers.
      if( val == 0 ) bOk = false;
      rng->end( val );
   }

   if( bOk && ! ops[2].isNil() )
   {
      bool bCreate = true;
      Expression* val = TreeStep::checkExpr(ops[2],bCreate);
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

   ctx->stackResult( pcount, SynClasses::collect( this, rng, __LINE__ ) );
}
TreeStep* SynClasses::ClassGenRange::createInstance() const { return new ExprRange; }


void SynClasses::ClassGenRef::op_create( VMContext* ctx, int pcount ) const
{
   static Class* symClass = Engine::instance()->symbolClass();
   static Class* exprClass = Engine::instance()->expressionClass();
   
   Item* params = *ctx->opcodeParams(pcount);
   Class* cls;
   void* data;
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
      ExprValue* expr = new ExprRef( sym );
      ctx->stackResult( pcount, SynClasses::collect( this, expr, __LINE__ ) );
   }
   else if( cls->isDerivedFrom( exprClass ) &&
      static_cast<Expression*>(data)->type() == Expression::e_trait_symbol )
   {
      //TODO:TreeStepInherit
      ExprSymbol* sym = static_cast<ExprSymbol*>( data );
      ExprValue* expr = new ExprRef( sym );
      ctx->stackResult( pcount, SynClasses::collect( this, expr, __LINE__ ) );
   }
   else {            
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Symbol|Sym") ) );
   }
}
TreeStep* SynClasses::ClassGenRef::createInstance() const { return new ExprRef; }

void SynClasses::ClassGenSym::op_create( VMContext* ctx, int pcount ) const
{
   static Class* symClass = Engine::instance()->symbolClass();
   
   Item* params = *ctx->opcodeParams(pcount);
   Class* cls;
   void* data;
   if( pcount < 1 || params->asClassInst(cls, data) || ! cls->isDerivedFrom(symClass) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Symbol") ) );
   }
   
   //TODO:TreeStepInherit
   Symbol* sym = static_cast<Symbol*>( data );
   ExprValue* expr = new ExprSymbol( sym );
   ctx->stackResult( pcount, SynClasses::collect( this, expr, __LINE__ ) );
}
TreeStep* SynClasses::ClassGenSym::createInstance() const { return new ExprSymbol; }

void SynClasses::ClassValue::op_create( VMContext* ctx, int pcount ) const
{
   if( pcount < 1 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("X") ) );
   }
   
   ExprValue* expr = new ExprValue( *ctx->opcodeParams(pcount) );
   ctx->stackResult( pcount, SynClasses::collect( this, expr, __LINE__ ) );
}

TreeStep* SynClasses::ClassValue::createInstance() const { return new ExprValue; }

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
FALCON_STANDARD_SYNCLASS_OP_CREATE( Select, StmtSelect, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Try, StmtTry, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( While, StmtWhile, zeroaryExprSet )


//=================================================================
// Trees
//
FALCON_STANDARD_SYNCLASS_OP_CREATE( RuleSynTree, RuleSynTree, zeroaryExprSet )
   
}

/* end of synclasses.cpp */
