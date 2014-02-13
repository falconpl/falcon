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
#include <falcon/stdsteps.h>
#include <falcon/stdhandlers.h>
#include <falcon/synclasses_id.h>
#include <falcon/stderrors.h>

#include <falcon/psteps/breakpoint.h>
#include <falcon/psteps/exprarray.h>
#include <falcon/psteps/expraccumulator.h>
#include <falcon/psteps/exprassign.h>
#include <falcon/psteps/exprautoclone.h>
#include <falcon/psteps/exprbitwise.h>
#include <falcon/psteps/exprcall.h>
#include <falcon/psteps/exprcase.h>
#include <falcon/psteps/exprep.h>
#include <falcon/psteps/exprclosure.h>
#include <falcon/psteps/exprcompare.h>
#include <falcon/psteps/exprcompose.h>
#include <falcon/psteps/exprdict.h>
#include <falcon/psteps/exprdot.h>
#include <falcon/psteps/expreeq.h>
#include <falcon/psteps/exprin.h>
#include <falcon/psteps/exprnotin.h>
#include <falcon/psteps/expriif.h>
#include <falcon/psteps/exprincdec.h>
#include <falcon/psteps/exprindex.h>
#include <falcon/psteps/exprinherit.h>
#include <falcon/psteps/exprinvoke.h>
#include <falcon/psteps/expristring.h>
#include <falcon/psteps/exprlit.h>
#include <falcon/psteps/exprlogic.h>
#include <falcon/psteps/exprmath.h>
#include <falcon/psteps/exprmultiunpack.h>
#include <falcon/psteps/exprnamed.h>
#include <falcon/psteps/exprneg.h>
#include <falcon/psteps/exproob.h>
#include <falcon/psteps/exprparentship.h>
#include <falcon/psteps/exprproto.h>
#include <falcon/psteps/exprpseudocall.h>
#include <falcon/psteps/exprrange.h>
#include <falcon/psteps/exprself.h>
#include <falcon/psteps/exprfself.h>
#include <falcon/psteps/exprinit.h>
#include <falcon/psteps/exprstarindex.h>
#include <falcon/psteps/exprstripol.h>
#include <falcon/psteps/exprsummon.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprtree.h>
#include <falcon/psteps/exprunpack.h>
#include <falcon/psteps/exprunquote.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/exprevalret.h>
#include <falcon/psteps/exprevalretexec.h>

#include <falcon/psteps/stmtbreak.h>
#include <falcon/psteps/stmtcontinue.h>
#include <falcon/psteps/stmtfastprint.h>
#include <falcon/psteps/stmtfastprintnl.h>
#include <falcon/psteps/stmtfor.h>
#include <falcon/psteps/stmtglobal.h>
#include <falcon/psteps/stmtif.h>
#include <falcon/psteps/stmtloop.h>
#include <falcon/psteps/stmtraise.h>
#include <falcon/psteps/stmtreturn.h>
#include <falcon/psteps/exprrule.h>
#include <falcon/psteps/stmtselect.h>
#include <falcon/psteps/stmtswitch.h>
#include <falcon/psteps/stmttry.h>
#include <falcon/psteps/stmtwhile.h>

#include <falcon/itemarray.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/symbol.h>
#include <falcon/storer.h>

namespace Falcon {
#undef FALCON_SYNCLASS_DECLARATOR_DECLARE
#define FALCON_SYNCLASS_DECLARATOR_APPLY

SynClasses::SynClasses( Class* classSynTree, Class* classStatement, Class* classExpr ):
   m_cls_SynTree( classSynTree ),
   m_cls_Statement( classStatement ),
   m_cls_Expression( classExpr ),
   m_dummy_end(0)
{
   m_cls_TreeStep = Engine::handlers()->treeStepClass();

   #include <falcon/synclasses_list.h>

   m_expr_ep->userFlags(FALCON_SYNCLASS_ID_EPEX);
   m_expr_invoke->userFlags(FALCON_SYNCLASS_ID_INVOKE);
   m_stmt_forto->userFlags(FALCON_SYNCLASS_ID_FORCLASSES);
   m_stmt_forin->userFlags(FALCON_SYNCLASS_ID_FORCLASSES);
   m_stmt_rule->userFlags(FALCON_SYNCLASS_ID_RULE);
   m_stmt_if->userFlags(FALCON_SYNCLASS_ID_ELSEHOST);
   m_stmt_select->userFlags(FALCON_SYNCLASS_ID_SELECT );
   m_stmt_switch->userFlags(FALCON_SYNCLASS_ID_SWITCH );
   m_stmt_try->userFlags(FALCON_SYNCLASS_ID_CATCHHOST);
   m_expr_call->userFlags(FALCON_SYNCLASS_ID_CALLFUNC);
   m_expr_pseudocall->userFlags(FALCON_SYNCLASS_ID_CALLFUNC);
   m_expr_assign->userFlags(FALCON_SYNCLASS_ID_ASSIGN);

   m_expr_tree->userFlags(FALCON_SYNCLASS_ID_TREE);
   m_st_rulest->userFlags(FALCON_SYNCLASS_ID_RULE_SYNTREE);

}

SynClasses::~SynClasses() {}
 
void SynClasses::subscribe( Engine* engine )
{
   #undef FALCON_SYNCLASS_DECLARATOR_APPLY
   #define FALCON_SYNCLASS_DECLARATOR_REGISTER
   #include <falcon/synclasses_list.h>
}
   

void SynClasses::varExprInsert( VMContext* ctx, int pcount, TreeStep* step, bool hasSel )
{
   Item* operands = ctx->opcodeParams(pcount);
   int count = 0;

   if( hasSel )
   {
      // the fisrt parameter is a selector.
      if( pcount == 0 )
      {
         // ... and is mandatory
         throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
                     .origin( ErrorParam::e_orig_runtime)
                     .extra( "Expression,..." ) );
      }

      bool bCreate = true;
      Expression* sel = TreeStep::checkExpr(operands[count++], bCreate);
      if( sel == 0 || ! step->selector(sel) )
      {
         if( bCreate )
         {
            delete sel;
         }

         throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
                     .origin( ErrorParam::e_orig_runtime)
                     .extra( String("Incompatible entity at ").N(count) ) );
      }
   }


   while( count < pcount )
   {
      bool bCreate = true;
      TreeStep* ts = TreeStep::checkExpr(operands[count++], bCreate);
      if( ts == 0 )
      {
         throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Incompatible entity at ").N(count) ) );
      }
      
      // larger than size, as we ++count before, but it's ok.
      if( ! step->insert(count, ts) )
      {
         // theoretically parented entities are not created, but...
         if ( bCreate ) delete ts;
         
         // params count 1 to N so we're ok to use count that has been ++
         throw new ParamError( ErrorParam( e_expr_assign, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("at ").N(count) ) );
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
      if( ! step->setNth(count, ts) )
      {
         delete step;
         if ( bCreate ) delete ts;
         
         if( ts->parent() )
            throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
               .origin( ErrorParam::e_orig_runtime)
               .extra( String("Already parented entity at ").N(count+1) ) );
         else
            throw new ParamError( ErrorParam( e_param_range, __LINE__, SRC )
                           .origin( ErrorParam::e_orig_runtime)
                           .extra( String("Index out of range") ) );

      }
      ++count;
   }
}

GCToken* SynClasses::collect( const Class* cls, TreeStep* earr, int line )
{
   (void) line;
   return FALCON_GC_STORE_SRCLINE( cls, earr, SRC, line );
}

//===========================================================
// The classes
//

#define FALCON_STANDARD_SYNCLASS_OP_CREATE_INIT( cls, exprcls, operation, init ) \
   void* SynClasses::Class## cls ::createInstance() const { init } \
   bool SynClasses::Class## cls ::op_init( VMContext* ctx, void* instance, int pcount ) const\
   {\
      exprcls* expr = static_cast<exprcls*>(instance); \
      expr->setInGC(); \
      operation ( ctx, pcount, expr ); \
      return false; \
   }\
   void SynClasses::Class##cls ::restore( VMContext* ctx, DataReader*dr) const \
   {\
      exprcls* ec = new exprcls; \
      try {\
         ctx->pushData( Item( this, ec) ); \
         m_parent->restore( ctx, dr ); \
      }catch(...) {\
         delete ec; \
         ctx->popData(); \
         throw; \
      }\
   }

#define FALCON_STANDARD_SYNCLASS_OP_CREATE( cls, exprcls, operation ) \
         FALCON_STANDARD_SYNCLASS_OP_CREATE_INIT(cls, exprcls, operation, return new exprcls;)


#define FALCON_STANDARD_SYNCLASS_OP_CREATE_SIMPLE( cls, exprcls, operation ) \
   void* SynClasses::Class## cls ::createInstance() const { return new exprcls; } \
   bool SynClasses::Class## cls ::op_init( VMContext* ctx, void* instance, int pcount ) const\
   {\
      exprcls* expr = static_cast<exprcls*>(instance); \
      expr->setInGC(); \
      operation ( ctx, pcount, expr ); \
      return false; \
   }\



FALCON_STANDARD_SYNCLASS_OP_CREATE( GenArray, ExprArray, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Accumulator, ExprAccumulator, ternaryExprSet)
FALCON_STANDARD_SYNCLASS_OP_CREATE( Assign, ExprAssign, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( BNot, ExprBNOT, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Call, ExprCall, varExprInsert_sel )
// Case -- specifically managed
FALCON_STANDARD_SYNCLASS_OP_CREATE( EP, ExprEP, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Invoke, ExprInvoke, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( StrIPol, ExprStrIPol, unaryExprSet )
// GenClosure --specifically managed

FALCON_STANDARD_SYNCLASS_OP_CREATE( LT, ExprLT, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( LE, ExprLE, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( GT, ExprGT, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( GE, ExprGE, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( EQ, ExprEQ, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( NE, ExprNE, binaryExprSet )
   
// GenDict --specifically managed
// DotAccess -- specifically managed
FALCON_STANDARD_SYNCLASS_OP_CREATE( EEQ, ExprEEQ, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( In, ExprIn, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Notin, ExprNotin, binaryExprSet )
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

// Functional
FALCON_STANDARD_SYNCLASS_OP_CREATE( Compose, ExprCompose, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Neg, ExprNeg, unaryExprSet )

// OOB
FALCON_STANDARD_SYNCLASS_OP_CREATE( Oob, ExprOob, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( DeOob, ExprDeoob, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( IsOob, ExprIsOob, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( XorOob, ExprXorOob, unaryExprSet )

// GenProto -- separated
// Pseudocall -- separated
// GenRange -- seperated
FALCON_STANDARD_SYNCLASS_OP_CREATE( Self, ExprSelf, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( FSelf, ExprFSelf, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Init, ExprInit, zeroaryExprSet )

FALCON_STANDARD_SYNCLASS_OP_CREATE( StarIndexAccess, ExprStarIndex, binaryExprSet )

// Sym -- separated
FALCON_STANDARD_SYNCLASS_OP_CREATE( Unpack, ExprUnpack, varExprInsert_sel )
FALCON_STANDARD_SYNCLASS_OP_CREATE( MUnpack, ExprMultiUnpack, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Unquote, ExprUnquote, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( EvalRet, ExprEvalRet, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( EvalRetExec, ExprEvalRetExec, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( EvalRetDoubt, ExprEvalRetDoubt, unaryExprSet )
// Value -- separated
   
//=================================================================
// Specific management
//

void*  SynClasses::ClassGenClosure::createInstance() const
{
   return new ExprClosure;
}
bool SynClasses::ClassGenClosure::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   ExprClosure* expr = static_cast<ExprClosure*>(instance);
   expr->setInGC();

   if( pcount < 1 || ! ctx->topData().isFunction() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Function") ) );
   }
   
   expr->function( ctx->topData().asFunction() );
   return false;
}
void SynClasses::ClassGenClosure::restore( VMContext* ctx, DataReader*dr ) const
{
   ExprClosure* expr = new ExprClosure;
   try {
      ctx->pushData( Item( this, new ExprClosure) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}
void SynClasses::ClassGenClosure::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
    ExprClosure* cl = static_cast<ExprClosure*>(instance);
    
    if( cl->function() != 0 ) {
       subItems.reserve(1);
       subItems.append( Item(cl->function()->handler(), cl->function() ) );
    }
}
void SynClasses::ClassGenClosure::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   if( subItems.length() > 0 ) {
      Function* func = static_cast<Function*>( subItems[0].asInst() );
      ExprClosure* cl = static_cast<ExprClosure*>(instance);
      cl->function(func);
   }
}


void* SynClasses::ClassGenDict::createInstance() const
{
   return new ExprDict;
}
bool SynClasses::ClassGenDict::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   ExprDict* expr = static_cast<ExprDict*>(instance);
   expr->setInGC();
   
   if( pcount % 2 != 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Need pair count of expressions") ) );
   }
   
   SynClasses:: varExprInsert( ctx, pcount, expr );
   return false;
}
void SynClasses::ClassGenDict::restore( VMContext* ctx, DataReader*dr ) const
{
   ExprDict* expr = new ExprDict;
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}


void* SynClasses::ClassDotAccess::createInstance() const
{
   return new ExprDot;
}
bool SynClasses::ClassDotAccess::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   if( (pcount < 2) || (! ctx->opcodeParams(pcount)[1].isString()) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Expression,S") ) );
   }
   
   ExprDot* expr = static_cast<ExprDot*>(instance);
   expr->setInGC();
   SynClasses::unaryExprSet( ctx, pcount, expr );
   expr->property( *ctx->opcodeParams(pcount)[1].asString() );
   return false;
}

void SynClasses::ClassDotAccess::store( VMContext* ctx, DataWriter*wr, void* instance ) const
{
   ExprDot* dot = static_cast<ExprDot*>( instance );
   wr->write(dot->property());
   m_parent->store( ctx, wr, dot );
}

void SynClasses::ClassDotAccess::restore( VMContext* ctx, DataReader*dr ) const
{
   String prop;
   dr->read( prop );   
   ExprDot* dot = new ExprDot;
   dot->property( prop );

   try {
      ctx->pushData( Item( this, dot ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete dot;
      throw;
   }
}
void SynClasses::ClassDotAccess::op_getProperty( VMContext* ctx, void* instance, const String& property )const
{
   if( property == "property" )
   {
      ExprDot* dot = static_cast<ExprDot*>( instance );
      ctx->topData() = FALCON_GC_HANDLE(new String(dot->property()));
      return;
   }
   m_parent->op_getProperty(ctx, instance, property);
}

void* SynClasses::ClassGenProto::createInstance() const
{       
   return new ExprProto;
}
bool SynClasses::ClassGenProto::op_init( VMContext* ctx, void* instance, int pcount ) const
{       
   // TODO -- parse a list of pairs string->expression
   return Class::op_init( ctx, instance, pcount );
}
void SynClasses::ClassGenProto::restore( VMContext* ctx, DataReader*dr ) const
{
   ExprProto* expr = new ExprProto;

   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}
void SynClasses::ClassGenProto::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ExprProto* ep = static_cast<ExprProto*>(instance);
   uint32 size = ep->size();

   subItems.reserve(size*2);
   for(uint32 i = 0; i < size; ++i )
   {
      const String& name = ep->nameAt(i);
      Expression* expr = ep->exprAt(i);

      subItems.append(Item(name.handler(), const_cast<String*>(&name)));
      subItems.append(Item(expr->handler(), expr));
   }
}
void SynClasses::ClassGenProto::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   fassert( subItems.length() % 2 == 0 );

   ExprProto* ep = static_cast<ExprProto*>(instance);

   for(uint32 i = 0; i < subItems.length(); i += 2 )
   {
      const Item& first = subItems[i];
      const Item& second = subItems[i+1];

      fassert( first.isString() );
      fassert( second.asClass()->isDerivedFrom(Engine::handlers()->treeStepClass()));

      ep->add(*first.asString(), static_cast<Expression*>(second.asInst()));
   }
}
void* SynClasses::ClassPseudoCall::createInstance() const
{       
   return new ExprPseudoCall;
}
bool SynClasses::ClassPseudoCall::op_init( VMContext* ctx, void* instance, int pcount ) const
{       
   // Pseudocall is abstract
   return Class::op_init( ctx, instance, pcount );
}
void SynClasses::ClassPseudoCall::restore( VMContext* ctx, DataReader*dr ) const
{
   fassert( "Not yet implemented" );
   ctx->pushData( Item( this, new ExprPseudoCall ) );
   m_parent->restore( ctx, dr );
}


void* SynClasses::ClassGenRange::createInstance() const
{       
   return new ExprRange;
}
bool SynClasses::ClassGenRange::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   if( pcount < 3 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Expression|Nil,Expression|Nil,Expression|Nil") ) );
   }
      
   ExprRange* rng = static_cast<ExprRange*>(instance);
   rng->setInGC();
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

   return false;
}
void SynClasses::ClassGenRange::restore( VMContext* ctx, DataReader*dr ) const
{
   ExprRange* expr = new ExprRange;
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}


void* SynClasses::ClassGenSym::createInstance() const
{
   return new ExprSymbol;
}
bool SynClasses::ClassGenSym::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   static Class* symClass = Engine::handlers()->symbolClass();

   ExprSymbol* expr = static_cast<ExprSymbol*>( instance );
   Item* params = ctx->opcodeParams(pcount);
   expr->setInGC();

   if( pcount >= 1 )
   {
      Class* cls;
      void* data;
      if( params->isString() )
      {
         expr->name(*params->asString());
         return false;
      }
      else if( params->asClassInst(cls, data) && cls->isDerivedFrom(symClass) )
      {
         const Symbol* sym = static_cast<Symbol*>(symClass->getParentData( cls, data ));
         expr->symbol( sym );
         return false;
      }
   }
      
   throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_runtime)
         .extra( String("Symbol|S") ) );
   //TODO:TreeStepInherit
}
void SynClasses::ClassGenSym::op_call(VMContext* ctx, int pcount, void* instance) const
{
   ExprSymbol* esym = static_cast<ExprSymbol*>(instance);
   const Symbol* sym = esym->symbol();

   Item* res = ctx->resolveSymbol(sym, false);
   if( res == 0 )
      ctx->stackResult(pcount+1, Item());
   else
      ctx->stackResult(pcount+1, *res);
}
void SynClasses::ClassGenSym::store( VMContext*, DataWriter* dw, void* instance ) const
{
   ExprSymbol* es = static_cast<ExprSymbol*>(instance);
   dw->write( es->line() );
   dw->write( es->chr() );
   dw->write( es->name() );
   dw->write( es->isPure() );

}
void SynClasses::ClassGenSym::restore( VMContext* ctx, DataReader*dr ) const
{
   int32 line, chr;
   String name;
   bool bIsPure;
   dr->read( line );
   dr->read( chr );
   dr->read( name );
   dr->read( bIsPure );

   ExprSymbol* es = new ExprSymbol;
   ctx->pushData( Item( this, es ) );
   es->decl( line, chr );
   es->setPure( bIsPure );
   const Symbol* sym = Engine::getSymbol( name );
   es->symbol( sym );
   sym->decref();
}
 
//==========================================
// Expr value
void* SynClasses::ClassValue::createInstance() const
{
   return new ExprValue;
}
bool SynClasses::ClassValue::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   if( pcount < 1 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("X") ) );
   }
   
   ExprValue* expr = static_cast<ExprValue*>(instance);
   expr->setInGC();
   expr->item( ctx->topData() );
   return false;
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

void SynClasses::ClassValue::restore( VMContext* ctx, DataReader*dr ) const
{
   ExprValue* expr = new ExprValue;
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}

//==========================================
// Expr value
void* SynClasses::ClassAutoClone::createInstance() const
{
   return new ExprValue;
}
bool SynClasses::ClassAutoClone::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   if( pcount < 1 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("X") ) );
   }

   ExprAutoClone* expr = static_cast<ExprAutoClone*>(instance);
   expr->setInGC();
   Class* cls;
   void* value;
   ctx->topData().forceClassInst(cls, value);
   expr->set( cls, cls->clone(value) );
   return false;
}

void SynClasses::ClassAutoClone::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ExprAutoClone* ev = static_cast<ExprAutoClone*>( instance );
   subItems.resize(1);
   if( ev->cloneHandler() != 0 )
   {
      subItems[0].setUser(ev->cloneHandler(), ev->cloneData());
   }
}

void SynClasses::ClassAutoClone::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   fassert(subItems.length() == 1);
   ExprAutoClone* ev = static_cast<ExprAutoClone*>( instance );
   if( subItems[0].isUser() )
   {
      ev->set( subItems[0].asClass(), subItems[0].asInst() );
   }
}

void SynClasses::ClassAutoClone::restore( VMContext* ctx, DataReader*dr ) const
{
   ExprAutoClone* expr = new ExprAutoClone;
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}

//==========================================
// Expr istring
void* SynClasses::ClassIString::createInstance() const
{
   return new ExprIString;
}
bool SynClasses::ClassIString::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   if( pcount < 1 || ! ctx->topData().isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("S") ) );
   }

   ExprIString* expr = static_cast<ExprIString*>(instance);
   expr->original( *ctx->topData().asString() );
   expr->setInGC();

   return false;
}
void SynClasses::ClassIString::store( VMContext* ctx, DataWriter* dw, void* instance ) const
{
   ExprIString* expr = static_cast<ExprIString*>(instance);
   dw->write(expr->original());
   m_parent->store(ctx, dw, instance);
}
void SynClasses::ClassIString::restore( VMContext* ctx, DataReader*dr ) const
{
   String orig;
   dr->read(orig);

   ExprIString* expr = new ExprIString(orig);
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );

   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}

//=================================================================
// Parentship / inheritance
//
void* SynClasses::ClassInherit::createInstance() const
{
   return new ExprInherit;
}
bool SynClasses::ClassInherit::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   static Class* clsExpr = Engine::handlers()->expressionClass();
      
   if( pcount < 1 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("C,...") ) );
   }
   
   Item* operands = ctx->opcodeParams( pcount );
   register Item* clsItem = operands;
   // is that really a class?
   if( ! clsItem->isClass() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("C,...") ) );
   }
   
   Class* theClass = static_cast<Class*>(clsItem->asInst());
   ExprInherit* inh = static_cast<ExprInherit*>(instance);
   inh->setInGC();

   inh->base( theClass );
   
   // and now, the init expressions
   for( int i = 1; i < pcount; ++i ) 
   {
      register Item* exprItem = operands + i;
      Class* cls;
      void* data;
      exprItem->forceClassInst(cls, data);
      if( ! cls->isDerivedFrom( clsExpr ) ) {
         throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Parameter ").N(i).A(" is not an expression") ) );
      }
      Expression* expr = static_cast<Expression*>( cls->getParentData(clsExpr, data) );
      inh->append(expr);
   }
 
   return false;
}
void SynClasses::ClassInherit::store( VMContext* ctx, DataWriter* wr, void* instance ) const
{
   ExprInherit* inh = static_cast<ExprInherit*>(instance);
   wr->write( inh->symbol()->name() );

   m_parent->store( ctx, wr, instance );
}
void SynClasses::ClassInherit::restore( VMContext* ctx, DataReader* dr ) const
{
   String name;
   dr->read( name );

   ExprInherit* expr = new ExprInherit(name);
   
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}



void* SynClasses::ClassParentship::createInstance() const
{
   return new ExprParentship;
}
bool SynClasses::ClassParentship::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   static Class* clsParent = 
               static_cast<Class*>( Engine::instance()
                     ->getMantra("Syn.Inherit", Mantra::e_c_class ) );
   fassert( clsParent != 0 );
   
   if( pcount == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("...") ) );
   }
   
   Item* operands = ctx->opcodeParams( pcount );
   ExprParentship* pship = static_cast<ExprParentship*>( instance );
   pship->setInGC();

   // and now, the init expressions
   for( int i = 0; i < pcount; ++i ) 
   {
      register Item* exprItem = operands + i;
      Class* cls;
      void* data;
      exprItem->forceClassInst(cls, data);
      if( ! cls->isDerivedFrom( clsParent ) ) {
         throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Parameter ").N(i).A(" is not an Inherit expression") ) );
      }
      Expression* expr = static_cast<Expression*>( cls->getParentData(clsParent, data) );
      pship->append(expr);
   }
 
   return false;
}
void SynClasses::ClassParentship::restore( VMContext* ctx, DataReader* dr ) const
{
   ExprParentship* expr = new ExprParentship;
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}

//=========================================
// Class Global
//

void* SynClasses::ClassGlobal::createInstance() const
{
   return new StmtGlobal;
}
bool SynClasses::ClassGlobal::op_init( VMContext* ctx, void* instance, int pCount ) const
{
   if( pCount == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
                        .extra("S|Symbol,..."));
   }

   StmtGlobal* stmt = static_cast<StmtGlobal *>(instance);
   stmt->setInGC();

   Item* params = ctx->opcodeParams(pCount);
   for( int i = 0; i < pCount; ++i )
   {
      Item* current = params+i;
      bool ok = true;
      bool done = false;
      if( current->isString() ) {
         ok = stmt->addSymbol( *current->asString() );
         done = true;
      }
      else if( current->isUser() )
      {
         Class* cls = 0;
         void* data = 0;
         current->asClassInst(cls, data);
         if( cls->typeID() == FLC_CLASS_ID_SYMBOL )
         {
            const Symbol* sym = static_cast<Symbol*>(data);
            // do not use the symbol directly.
            ok = stmt->addSymbol( sym->name() );
            done = true;
         }
      }

      if( ! done ) {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
                  .extra("S|Symbol,..."));
      }

      if( ! ok ) {
         throw ctx->runtimeError( e_global_again );
      }
   }

   return false;
}
void SynClasses::ClassGlobal::store( VMContext* ctx, DataWriter* wr, void* instance ) const
{
   StmtGlobal* g = static_cast<StmtGlobal*>(instance);
   g->store(wr);
   m_parent->store( ctx, wr, instance );
}
void SynClasses::ClassGlobal::restore( VMContext* ctx, DataReader* dr ) const
{
   StmtGlobal* stmt = new StmtGlobal;

   try {
      stmt->restore( dr );
      ctx->pushData( Item( this, stmt ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete stmt;
      throw;
   }
}

//=========================================
// Class LIT
//

void* SynClasses::ClassLit::createInstance() const
{
   // lit cannot be invoked by constructor.
   return 0;
}
bool SynClasses::ClassLit::op_init( VMContext* , void* instance, int ) const
{
   ExprLit* expr = static_cast<ExprLit*>(instance);
   expr->setInGC();
   // TODO
   return true;
}
void SynClasses::ClassLit::restore( VMContext* ctx, DataReader* dr ) const
{
   ExprLit* expr = new ExprLit;
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}
void SynClasses::ClassLit::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ExprLit* lit = static_cast<ExprLit*>(instance);
   subItems.append( Item( lit->child()->handler(), lit->child()) );

   for( uint32 i = 0; i < lit->unquotedCount(); ++ i ) {
      Expression* unquoted = lit->unquoted(i);
      subItems.append( Item(unquoted->handler(), unquoted) );
   }
}
void SynClasses::ClassLit::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ExprLit* lit = static_cast<ExprLit*>(instance);
   if( subItems.length() >= 1 ) {
      fassert(subItems.at(0).type() == FLC_CLASS_ID_TREESTEP );
      lit->setChild( static_cast<TreeStep*>(subItems.at(0).asInst()) );
   }

   for( uint32 i = 1; i < subItems.length(); ++ i ) {
      Item& item = subItems[i];
      fassert(subItems.at(i).type() == FLC_CLASS_ID_TREESTEP );
      lit->registerUnquote( static_cast<Expression*>(item.asInst()) );
   }
}
//=========================================
// Class Tree
//
void* SynClasses::ClassTree::createInstance() const
{
   return new ExprTree;
}
void SynClasses::ClassTree::op_call(VMContext* ctx, int pcount, void* instance) const
{
   ExprTree* tree = static_cast<ExprTree*>(instance);
   TreeStep* child = tree->child();

   // TODO: really need to check for childhood?
   if( child != 0 ) {
      SymbolMap* st = tree->parameters();
      // We must always push a local frame, also with st == 0
      ctx->addLocalFrame( st, pcount );
      ctx->pushCode( child );
   }
   else {
      ctx->stackResult(pcount+1,Item());
   }
}
bool SynClasses::ClassTree::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   // the first parameter must be a tree step.
   Item* i_tree = ctx->opcodeParams(pcount);
   if ( pcount == 0 || (i_tree->type() != FLC_CLASS_ID_TREESTEP) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime )
            .extra( "TreeStep,[S]..." ) );
   }

   // put in the treestep
   ExprTree* self = static_cast<ExprTree*>(instance);
   self->setInGC();
   TreeStep* ts = static_cast<TreeStep*>(i_tree->asInst());
   self->setChild(ts);

   // And now the parameters.
   for( int i = 1; i < pcount; ++ i) {
      Item* param = i_tree+i;
      if( ! param->isString() )
      {
         throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("Parameter ").N(i).A(" is not an Inherit expression") ) );

      }
      self->addParam(*param->asString());
   }

   // we already managed.
   return false;
}
void SynClasses::ClassTree::store( VMContext* ctx, DataWriter* wr, void* instance ) const
{
   ExprTree* tree = static_cast<ExprTree*>(instance);

   m_parent->store( ctx, wr, tree );

   wr->write( tree->isEta() );
   bool vmap = tree->parameters() != 0;
   wr->write( vmap );
   if( vmap ) {
      tree->parameters()->store( wr );
   }
}
void SynClasses::ClassTree::restore( VMContext* ctx, DataReader* dr ) const
{
   ExprTree* expr = new ExprTree;
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
      bool hasVmap = false, isEta = false;
      dr->read( isEta );
      dr->read( hasVmap );

      expr->setEta(isEta);
      if( hasVmap ) {
         expr->setParameters( new SymbolMap );
         expr->parameters()->restore( dr );
      }
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}
void SynClasses::ClassTree::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ExprTree* tree = static_cast<ExprTree*>(instance);
   if( tree->child() != 0 ) {
      subItems.append( Item( tree->child()->handler(), tree->child()) );
   }
}
void SynClasses::ClassTree::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ExprTree* tree = static_cast<ExprTree*>(instance);
   if( subItems.length() >= 1 && !subItems[0].isNil() ) {
      tree->setChild( static_cast<TreeStep*>(subItems.at(0).asInst()) );
   }
}


void*  SynClasses::ClassCase::createInstance() const
{
   return new ExprCase;
}
bool SynClasses::ClassCase::op_init( VMContext* ctx, void* instance, int pCount ) const
{
   ExprCase* expr = static_cast<ExprCase*>(instance);
   expr->setInGC();

   Item* ptr = ctx->opcodeParams(pCount);
   for( int n = 0; n < pCount; ++ n )
   {
      if( ! expr->addEntry(ptr[n]) )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("Not a valid case entry") );
      }
   }

   return false;
}
void SynClasses::ClassCase::op_call(VMContext* ctx, int pcount, void* instance) const
{
   if ( pcount == 0 )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra( "X" ) );
   }

   ExprCase* expr = static_cast<ExprCase*>(instance);
   bool tof = expr->verify(ctx->opcodeParam(pcount));
   ctx->stackResult(pcount+1,Item().setBoolean(tof));
}
void SynClasses::ClassCase::store( VMContext* ctx, DataWriter* dw, void* instance ) const
{
   ExprCase* cls = static_cast<ExprCase*>(instance);
   m_parent->store( ctx, dw, instance );
   cls->store(dw);
}
void SynClasses::ClassCase::restore( VMContext* ctx, DataReader* dr ) const
{
   ExprCase* expr = new ExprCase;
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
      expr->restore(dr);
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}
void SynClasses::ClassCase::flatten( VMContext*, ItemArray& arr, void* instance ) const
{
   ExprCase* expr = static_cast<ExprCase*>(instance);
   expr->flatten(arr);
}

void SynClasses::ClassCase::unflatten( VMContext*, ItemArray& arr, void* instance) const
{
   ExprCase* expr = static_cast<ExprCase*>(instance);
   expr->unflatten(arr);
}

//=========================================
// Class Summon e OptSummon
//

void* SynClasses::ClassSummon::createInstance() const
{
   // lit cannot be invoked by constructor.
   return new ExprSummon;
}
bool SynClasses::ClassSummon::op_init( VMContext* , void* instance, int ) const
{
   ExprSummon* expr = static_cast<ExprSummon*>(instance);
   expr->setInGC();
   // TODO
   return true;
}
void SynClasses::ClassSummon::store( VMContext* ctx, DataWriter* dw, void* instance ) const
{
   ExprSummon* expr = static_cast<ExprSummon*>( instance );
   dw->write( expr->message() );
   m_parent->store( ctx, dw, instance );
}
void SynClasses::ClassSummon::op_getProperty( VMContext* ctx, void* instance, const String& property )const
{
   if( property == "message" )
   {
      ExprSummon* dot = static_cast<ExprSummon*>( instance );
      ctx->topData() = FALCON_GC_HANDLE(new String(dot->message()));
      return;
   }
   m_parent->op_getProperty(ctx, instance, property);
}
void SynClasses::ClassSummon::restore( VMContext* ctx, DataReader* dr ) const
{
   String message;
   dr->read(message);

   ExprSummon* expr = new ExprSummon;
   expr->message(message);
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}

void* SynClasses::ClassOptSummon::createInstance() const
{
   // lit cannot be invoked by constructor.
   return new ExprOptSummon;
}
bool SynClasses::ClassOptSummon::op_init( VMContext* , void* instance, int ) const
{
   ExprOptSummon* expr = static_cast<ExprOptSummon*>(instance);
   expr->setInGC();
   // TODO
   return true;
}
void SynClasses::ClassOptSummon::store( VMContext* ctx, DataWriter* dw, void* instance ) const
{
   ExprOptSummon* expr = static_cast<ExprOptSummon*>( instance );
   dw->write( expr->message() );
   m_parent->store( ctx, dw, instance );
}
void SynClasses::ClassOptSummon::op_getProperty( VMContext* ctx, void* instance, const String& property )const
{
   if( property == "message" )
   {
      ExprOptSummon* dot = static_cast<ExprOptSummon*>( instance );
      ctx->topData() = FALCON_GC_HANDLE(new String(dot->message()));
      return;
   }
   m_parent->op_getProperty(ctx, instance, property);
}
void SynClasses::ClassOptSummon::restore( VMContext* ctx, DataReader* dr ) const
{
   String message;
   dr->read(message);

   ExprOptSummon* expr = new ExprOptSummon;
   expr->message(message);
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}

//=================================================================
// Statements
//
static void init_selector_and_rest( VMContext* ctx, int pcount, Statement* step, bool bAcceptNil )
{
   if( pcount < 1 )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("Symbol|Expression,..."));
   }

   Item* params = ctx->opcodeParams(pcount);
   if( params->type() == FLC_CLASS_ID_SYMBOL )
   {
      if ( ! step->selector(new ExprSymbol(static_cast<Symbol*>(params->asInst()))) )
      {
         throw new ParamError( ErrorParam( e_expr_assign, __LINE__, SRC )
               .origin( ErrorParam::e_orig_runtime)
               .extra( String("at 0") ) );
      }
   }
   else if( params->type() == FLC_CLASS_ID_TREESTEP && static_cast<TreeStep*>(params->asInst())->category() == TreeStep::e_cat_expression)
   {
      if ( ! step->selector(static_cast<Expression*>(params->asInst())) )
      {
         throw new ParamError( ErrorParam( e_expr_assign, __LINE__, SRC )
               .origin( ErrorParam::e_orig_runtime)
               .extra( String("at 0") ) );
      }
   }
   else if( params->isNil() && bAcceptNil )
   {
      // this is ok, let it be nil.
   }
   else
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("Symbol|Expression,..."));
   }

   for( int count = 1; count < pcount; ++count )
   {
      Item* expr = params + count;
      if( expr->type() != FLC_CLASS_ID_TREESTEP )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("Not a TreeStep in body"));
      }

      TreeStep* child = static_cast<TreeStep*>(expr->asInst());
      if( ! step->append( child ) )
      {
         throw new ParamError( ErrorParam( e_expr_assign, __LINE__, SRC )
                        .origin( ErrorParam::e_orig_runtime)
                        .extra( String("at ").N(count)) );
      }
   }
}

static void init_while( VMContext* ctx, int pcount, StmtWhile* step )
{
   init_selector_and_rest( ctx, pcount, step, false );
}

static void init_switch( VMContext* ctx, int pcount, SwitchlikeStatement* step )
{
   init_selector_and_rest( ctx, pcount, step, false );
}

static void init_loop( VMContext* ctx, int pcount, StmtWhile* step )
{
   init_selector_and_rest( ctx, pcount, step, true );
}


static void init_try( VMContext* ctx, int pcount, Statement* step )
{
   if( pcount < 1 )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("SynTree,..."));
   }

   Item* params = ctx->opcodeParams(pcount);
   for( int i = 0; i < pcount; ++i )
   {
      const Item& par = params[i];
      if( par.type() == FLC_CLASS_ID_TREESTEP )
      {
         TreeStep* ts = static_cast<TreeStep*>(par.asInst());
         if( step->setNth(i,ts ) )
         {
            // ok, skip the throw...
            continue;
         }
      }
      throw new ParamError( ErrorParam( e_expr_assign, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("at 0") ) );
   }
}


static void init_generic_multistmt( VMContext* ctx, int pcount, TreeStep* step )
{
   Item* params = ctx->opcodeParams(pcount);
   for( int count = 0; count < pcount; ++count )
   {
      Item* expr = params + count;
      // Warning, we're converting prior to know the type,
      // but we'll check type() that is valid for all PSteps.
      TreeStep* child = static_cast<TreeStep*>(expr->asInst());
      if( expr->type() != FLC_CLASS_ID_TREESTEP )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra(String("Parameter ").N(count).A( " not accepted")));
      }

      if( ! step->append( child ) )
      {
         throw new ParamError( ErrorParam( e_expr_assign, __LINE__, SRC )
                        .origin( ErrorParam::e_orig_runtime)
                        .extra( String("at ").N(count)) );
      }
   }
}

static void init_for_in( VMContext* ctx, int pCount, TreeStep* step )
{
   Item* params = ctx->opcodeParams(pCount);
   StmtForIn* in = static_cast<StmtForIn*>(step);
   if(
          ! in->setTargetFromParam(    pCount > 0 ? params + 0 : 0 )
       || ! in->setSelectorFromParam(  pCount > 1 ? params + 1 : 0 )
       || ! in->setBodyFromParam(      pCount > 2 ? params + 2 : 0 )
       || ! in->setForFirstFromParam(  pCount > 3 ? params + 3 : 0 )
       || ! in->setForMiddleFromParam( pCount > 4 ? params + 4 : 0 )
       || ! in->setForLastFromParam(   pCount > 5 ? params + 5 : 0 )
   )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_params,
               .extra(String("Symbol|A,Expression,...")));
   }
}

static void init_for_to( VMContext* ctx, int pCount, TreeStep* step )
{
   Item* params = ctx->opcodeParams(pCount);

   StmtForTo* in = static_cast<StmtForTo*>(step);
   if(
          ! in->setTargetFromParam(    pCount > 0 ? params + 0 : 0 )
       || ! in->setStartExprFromParam( pCount > 1 ? params + 1 : 0 )
       || ! in->setEndExprFromParam(   pCount > 2 ? params + 2 : 0 )
       || ! in->setStepExprFromParam(  pCount > 3 ? params + 3 : 0 )
       || ! in->setBodyFromParam(      pCount > 4 ? params + 4 : 0 )
       || ! in->setForFirstFromParam(  pCount > 5 ? params + 5 : 0 )
       || ! in->setForMiddleFromParam( pCount > 6 ? params + 6 : 0 )
       || ! in->setForLastFromParam(   pCount > 7 ? params + 7 : 0 )
   )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_params,
               .extra(String("Symbol,Expression,Expression,...")));
   }
}


FALCON_STANDARD_SYNCLASS_OP_CREATE( Break, StmtBreak, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Breakpoint, Breakpoint, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Continue, StmtContinue, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Cut, StmtCut, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Doubt, StmtDoubt, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( FastPrint, StmtFastPrint, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE( FastPrintNL, StmtFastPrintNL, varExprInsert)
FALCON_STANDARD_SYNCLASS_OP_CREATE_SIMPLE( ForIn, StmtForIn, init_for_in )
FALCON_STANDARD_SYNCLASS_OP_CREATE( ForTo, StmtForTo, init_for_to )
FALCON_STANDARD_SYNCLASS_OP_CREATE( If, StmtIf, init_generic_multistmt )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Loop, StmtLoop, init_loop )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Raise, StmtRaise, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE_SIMPLE( Return, StmtReturn, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Rule, ExprRule, init_generic_multistmt )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Select, StmtSelect, init_switch )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Switch, StmtSwitch, init_switch )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Try, StmtTry, init_try )
FALCON_STANDARD_SYNCLASS_OP_CREATE( While, StmtWhile, init_while )

//=================================================================
// Statements
//


void SynClasses::ClassReturn::store( VMContext* ctx, DataWriter*wr, void* instance ) const
{
   StmtReturn* ret = static_cast<StmtReturn*>( instance );
   wr->write(ret->hasDoubt());
   wr->write(ret->hasEval());
   wr->write(ret->hasBreak());
   m_parent->store( ctx, wr, ret );
}

void SynClasses::ClassReturn::restore( VMContext* ctx, DataReader*dr ) const
{
   bool bHasDoubt;
   bool bHasEval;
   bool bHasBreak;

   dr->read( bHasDoubt );
   dr->read( bHasEval );
   dr->read( bHasBreak );


   StmtReturn* expr = new StmtReturn;
   expr->hasDoubt(bHasDoubt);
   expr->hasEval(bHasEval);
   expr->hasBreak(bHasBreak);

   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}


void SynClasses::ClassForIn::store( VMContext* ctx, DataWriter*wr, void* instance ) const
{
   StmtForIn* forin = static_cast<StmtForIn*>( instance );
   uint32 count = forin->paramCount();
   wr->write(count);
   for( uint32 i = 0; i < count; ++i )
   {
      const Symbol* tgt = forin->param(i);
      wr->write( tgt->name() );
   }
   m_parent->store( ctx, wr, forin );
}
void SynClasses::ClassForIn::restore( VMContext* ctx, DataReader*dr ) const
{
   uint32 count;
   dr->read( count );
   StmtForIn* expr = new StmtForIn;

   try {
      ctx->pushData( Item( this, expr ) );
      for( uint32 i = 0; i < count; ++i )
      {
         String name;

         dr->read( name );

         const Symbol* sym = Engine::getSymbol(name);
         expr->addParameter(sym);
      }

      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
}

void SynClasses::ClassForTo::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   static Class* clsSym = Engine::handlers()->symbolClass();

   StmtForTo* stmt = static_cast<StmtForTo*>(instance);
   TRACE1( "SynClasses::ClassForTo::flatten %s", stmt->describe().c_ize() );
   subItems.resize(8);
   if ( stmt->target() != 0 ) { subItems[0] = Item( clsSym, stmt->target() ); }
   if ( stmt->startExpr() != 0 ) { subItems[1] = Item( stmt->startExpr()->handler(), stmt->startExpr() ); }
   if ( stmt->endExpr() != 0 ) { subItems[2] = Item( stmt->endExpr()->handler(), stmt->endExpr() ); }
   if ( stmt->stepExpr() != 0 ) { subItems[3] = Item( stmt->stepExpr()->handler(), stmt->stepExpr() ); }
   if ( stmt->body() != 0 ) { subItems[4] = Item( stmt->body()->handler(), stmt->body() ); }
   if ( stmt->forFirst() != 0 ) { subItems[5] = Item( stmt->forFirst()->handler(), stmt->forFirst() ); }
   if ( stmt->forLast() != 0 ) { subItems[6] = Item( stmt->forLast()->handler(), stmt->forLast() ); }
   if ( stmt->forMiddle() != 0 ) { subItems[7] = Item( stmt->forMiddle()->handler(), stmt->forMiddle() ); }
}
void SynClasses::ClassForTo::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   StmtForTo* stmt = static_cast<StmtForTo*>(instance);
   if( subItems.length() == 8 )
   {
      MESSAGE( "SynClasses::ClassForTo::unflatten -- correct subItems size.");
      if( ! subItems[0].isNil() ) stmt->target( static_cast<Symbol*>(subItems[0].asInst()) );
      if( ! subItems[1].isNil() ) stmt->startExpr( static_cast<Expression*>( subItems[1].asInst() ) );
      if( ! subItems[2].isNil() ) stmt->endExpr( static_cast<Expression*>( subItems[2].asInst() ) );
      if( ! subItems[3].isNil() ) stmt->stepExpr( static_cast<Expression*>( subItems[3].asInst() ) );
      if( ! subItems[4].isNil() ) stmt->body( static_cast<SynTree*>(subItems[4].asInst()) );
      if( ! subItems[5].isNil() ) stmt->forFirst( static_cast<SynTree*>(subItems[5].asInst()) );
      if( ! subItems[6].isNil() ) stmt->forLast( static_cast<SynTree*>(subItems[6].asInst()) );
      if( ! subItems[7].isNil() ) stmt->forMiddle( static_cast<SynTree*>(subItems[7].asInst()) );
   }
   TRACE1( "SynClasses::ClassForTo::unflatten %s", stmt->describe().c_ize() );
}

//=================================================================
// Trees
//
FALCON_STANDARD_SYNCLASS_OP_CREATE( RuleSynTree, RuleSynTree, init_generic_multistmt )
   
}

/* end of synclasses.cpp */
