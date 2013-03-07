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
#include <falcon/derivedfrom.h>
#include <falcon/stdhandlers.h>

#include <falcon/synclasses_id.h>

#include <falcon/errors/paramerror.h>

#include <falcon/psteps/breakpoint.h>
#include <falcon/psteps/exprarray.h>
#include <falcon/psteps/exprassign.h>
#include <falcon/psteps/exprbitwise.h>
#include <falcon/psteps/exprcall.h>
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

#include <falcon/classes/classexpression.h>
#include <falcon/storer.h>

namespace Falcon {
#undef FALCON_SYNCLASS_DECLARATOR_DECLARE
#define FALCON_SYNCLASS_DECLARATOR_APPLY

SynClasses::SynClasses( Class* classSynTree, Class* classStatement, Class* classExpr ):
   m_cls_st( classSynTree ),
   m_cls_stmt( classStatement ),
   m_cls_expr( classExpr ),
   m_dummy_end(0)
{
   m_cls_treestep = Engine::handlers()->treeStepClass();

   #include <falcon/synclasses_list.h>

   m_expr_ep->userFlags(FALCON_SYNCLASS_ID_EPEX);
   m_expr_invoke->userFlags(FALCON_SYNCLASS_ID_INVOKE);
   m_stmt_forto->userFlags(FALCON_SYNCLASS_ID_FORCLASSES);
   m_stmt_forin->userFlags(FALCON_SYNCLASS_ID_FORCLASSES);
   m_stmt_rule->userFlags(FALCON_SYNCLASS_ID_RULE);
   m_stmt_if->userFlags(FALCON_SYNCLASS_ID_ELSEHOST);
   m_stmt_select->userFlags(FALCON_SYNCLASS_ID_CASEHOST );
   m_stmt_switch->userFlags(FALCON_SYNCLASS_ID_SWITCH );
   m_stmt_try->userFlags(FALCON_SYNCLASS_ID_CATCHHOST);
   m_expr_call->userFlags(FALCON_SYNCLASS_ID_CALLFUNC);
   m_expr_pseudocall->userFlags(FALCON_SYNCLASS_ID_CALLFUNC);

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
      if( ! step->setNth(count, ts) )
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
   line = line;
   return FALCON_GC_STORE_SRCLINE( cls, earr, SRC, line );
}

//===========================================================
// The classes
//

#define FALCON_STANDARD_SYNCLASS_OP_CREATE( cls, exprcls, operation ) \
   void* SynClasses::Class## cls ::createInstance() const { return new exprcls; } \
   bool SynClasses::Class## cls ::op_init( VMContext* ctx, void* instance, int pcount ) const\
   {\
      exprcls* expr = static_cast<exprcls*>(instance); \
      SynClasses::operation ( ctx, pcount, expr ); \
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

#define FALCON_STANDARD_SYNCLASS_OP_CREATE_EX( cls, exprcls, operation ) \
   void* SynClasses::Class## cls ::createInstance() const { return new exprcls; } \
   bool SynClasses::Class## cls ::op_init( VMContext* ctx, void* instance, int pcount ) const\
   {\
      exprcls* expr = static_cast<exprcls*>(instance); \
      SynClasses::operation ( ctx, pcount, expr ); \
      return false; \
   }\



FALCON_STANDARD_SYNCLASS_OP_CREATE( GenArray, ExprArray, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Assign, ExprAssign, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( BNot, ExprBNOT, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Call, ExprCall, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE( EP, ExprEP, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Invoke, ExprInvoke, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( StrIPol, ExprStrIPol, unaryExprSet )
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
FALCON_STANDARD_SYNCLASS_OP_CREATE( In, ExprIn, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Notin, ExprNotin, binaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( IIF, ExprIIF, ternaryExprSet )
// TODO: Manage lit
//FALCON_STANDARD_SYNCLASS_OP_CREATE( Lit, ExprLit, unaryExprSet )

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

// MUnpack -- separated
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
// Unpack -- separated
FALCON_STANDARD_SYNCLASS_OP_CREATE( Unquote, ExprUnquote, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( EvalRet, ExprEvalRet, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( EvalRetExec, ExprEvalRetExec, unaryExprSet )
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


void* SynClasses::ClassMUnpack::createInstance() const
{       
   return new ExprMultiUnpack;
}

bool SynClasses::ClassMUnpack::op_init( VMContext* ctx, void* instance, int pcount ) const
{       
   // TODO -- parse a list of pairs symbol->expression
   return Class::op_init( ctx, instance, pcount );
}
void SynClasses::ClassMUnpack::store(Falcon::VMContext* ctx, Falcon::DataWriter* stream, void* inst ) const
{
   ExprMultiUnpack* expr = static_cast<ExprMultiUnpack*>(inst);
   stream->write(expr->isTop());
   m_parent->store( ctx, stream, inst );
}
void SynClasses::ClassMUnpack::restore( VMContext* ctx, DataReader*dr ) const
{
   bool isTop;
   dr->read(isTop);

   ExprMultiUnpack* expr = new ExprMultiUnpack(isTop);
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
void SynClasses::ClassMUnpack::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   static Class* symClass = Engine::handlers()->symbolClass();
   ExprMultiUnpack* expr = static_cast<ExprMultiUnpack*>( instance );

   uint32 count = expr->targetCount();
   subItems.resize(count*2);
   for( uint32 i = 0; i < count; ++i )
   {
      Symbol* assignand = expr->getAssignand(i);
      Expression* assignee = expr->getAssignee(i);
      subItems[i*2].setUser(symClass, assignand );
      subItems[i*2+1].setUser(assignee->handler(), assignee);
   }
}

void SynClasses::ClassMUnpack::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ExprMultiUnpack* expr = static_cast<ExprMultiUnpack*>( instance );
   fassert(subItems.length() % 2 == 0);

   uint32 count = 0;
   while( count < subItems.length() )
   {
      Symbol* sym = static_cast<Symbol*>(subItems[count++].asInst());
      Expression* assign = static_cast<Expression*>(subItems[count++].asInst());
      expr->addAssignment( sym, assign );
   }
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
   // TODO -- parse a single pseudofunction and a list of parameters.
   return Class::op_init( ctx, instance, pcount );
}
void SynClasses::ClassPseudoCall::restore( VMContext* ctx, DataReader*dr ) const
{
   fassert( "Not yet implemented" );
   ctx->pushData( Item( this, new ExprPseudoCall ) );
   m_parent->restore( ctx, dr );
}

void* SynClasses::ClassUnpack::createInstance() const
{       
   return new ExprUnpack;
}
bool SynClasses::ClassUnpack::op_init( VMContext* ctx, void* instance, int pcount ) const
{       
   // TODO -- parse a list of pairs symbol, + 1 terminal expression
   return Class::op_init( ctx, instance, pcount );
}
void SynClasses::ClassUnpack::store(Falcon::VMContext* ctx, Falcon::DataWriter* stream, void* instance) const
{
   ExprUnpack* expr = static_cast<ExprUnpack*>( instance );
   uint32 count = expr->targetCount();
   stream->write( count );
   for( uint32 i = 0; i < count; ++i )
   {
      Symbol* sym = expr->getAssignand(i);
      stream->write( sym->name() );
   }

   m_parent->store( ctx, stream, instance );
}
void SynClasses::ClassUnpack::restore( VMContext* ctx, DataReader*dr ) const
{
   ExprUnpack* expr = new ExprUnpack;
   try {
      uint32 count;
      dr->read( count );

      for( uint32 i = 0; i < count; ++i )
      {
         String name;
         dr->read(name);

         Symbol* sym = Engine::getSymbol( name );
         expr->addAssignand(sym);
      }

      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
   }
   catch(...) {
      ctx->popData();
      delete expr;
      throw;
   }
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
         Symbol* sym = static_cast<Symbol*>(symClass->getParentData( cls, data ));
         expr->symbol( sym );
         return false;
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
   Symbol* sym = Engine::getSymbol( name );
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
      inh->add(expr);
   }
 
   return false;
}

void SynClasses::ClassInherit::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   ExprInherit* inh = static_cast<ExprInherit*>( instance );
   Storer* storer = ctx->getTopStorer();
   
   subItems.resize(1+ inh->arity() );
   Class* baseClass = inh->base();
   if( baseClass == 0 )
   {
      // should not happen, but... keep the item nil.
      return;
   }
   
   // to decide how to flatten a class, we need to know if we're flattening our module.
   if( storer != 0 && storer->topData() == baseClass->module() 
      // Yep, we're storing the module, so  we're not forced to store external classes
      && ! inh->hadRequirement()
      )
   {
      // so, it's a module and we had a requirement. We're not storing this at all.
      return;
   }
   
   // in all the other cases, properly store the class.
   subItems[0].setUser( baseClass->handler(), baseClass );
   for( int i = 0; i < inh->arity(); ++i )
   {
      TreeStep* expr = inh->nth( i );
      subItems[i+1].setUser( expr->handler(), expr );
   }
}
void SynClasses::ClassInherit::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   fassert(subItems.length() >= 1);
   const Item& item = subItems[0];
   if( item.isNil() )
   {
      return;
   }
   
   ExprInherit* inherit = static_cast<ExprInherit*>( instance );
   Class* cls = static_cast<Class*>(item.asInst());
   inherit->base( cls );
   for( int i = 1; i < (int) subItems.length(); ++i )
   {
      Expression* expr = static_cast<Expression*>( subItems[i].asInst() );
      inherit->add(expr);
   }
}
void SynClasses::ClassInherit::store( VMContext* ctx, DataWriter* wr, void* instance ) const
{
   ExprInherit* inh = static_cast<ExprInherit*>(instance);
   wr->write( inh->name() );
   wr->write( inh->hadRequirement() );
   m_parent->store( ctx, wr, instance );
}
void SynClasses::ClassInherit::restore( VMContext* ctx, DataReader* dr ) const
{
   String name;
   bool bHadReq;
   dr->read( name );
   dr->read( bHadReq );

   ExprInherit* expr = new ExprInherit(name);
   expr->hadRequirement( bHadReq );
   
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
                     ->getMantra("Inherit", Mantra::e_c_class ) );
   fassert( clsParent != 0 );
   
   if( pcount == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( String("...") ) );
   }
   
   Item* operands = ctx->opcodeParams( pcount );
   ExprParentship* pship = static_cast<ExprParentship*>( instance );
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
      pship->add(expr);
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
         if( cls->userFlags() == FALCON_SYNCLASS_ID_SYMBOL )
         {
            Symbol* sym = static_cast<Symbol*>(data);
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
bool SynClasses::ClassLit::op_init( VMContext* , void* , int ) const
{
   // Let the caller to remove the params
   return false;
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
      lit->setChild( static_cast<Expression*>(subItems.at(0).asInst()) );
   }

   for( uint32 i = 1; i < subItems.length(); ++ i ) {
      Item& item = subItems[i];
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
      VarMap* st = tree->varmap();
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
   bool vmap = tree->varmap() != 0;

   m_parent->store( ctx, wr, tree );

   wr->write( vmap );
   if( vmap ) {
      tree->varmap()->store( wr );
   }
}
void SynClasses::ClassTree::restore( VMContext* ctx, DataReader* dr ) const
{
   ExprTree* expr = new ExprTree;
   try {
      ctx->pushData( Item( this, expr ) );
      m_parent->restore( ctx, dr );
      bool hasVmap;
      dr->read( hasVmap );

      if( hasVmap ) {
         expr->setVarMap( new VarMap );
         expr->varmap()->restore( dr );
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
   
//=================================================================
// Statements
//

FALCON_STANDARD_SYNCLASS_OP_CREATE( Break, StmtBreak, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Breakpoint, Breakpoint, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Continue, StmtContinue, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Cut, StmtCut, zeroaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Doubt, StmtDoubt, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE_EX( FastPrint, StmtFastPrint, varExprInsert )
FALCON_STANDARD_SYNCLASS_OP_CREATE_EX( ForIn, StmtForIn, zeroaryExprSet ) //
FALCON_STANDARD_SYNCLASS_OP_CREATE( ForTo, StmtForTo, zeroaryExprSet ) //
FALCON_STANDARD_SYNCLASS_OP_CREATE( If, StmtIf, zeroaryExprSet )   //
FALCON_STANDARD_SYNCLASS_OP_CREATE( Loop, StmtLoop, zeroaryExprSet ) //
FALCON_STANDARD_SYNCLASS_OP_CREATE( Raise, StmtRaise, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE_EX( Return, StmtReturn, unaryExprSet )
FALCON_STANDARD_SYNCLASS_OP_CREATE( Rule, ExprRule, zeroaryExprSet ) //
FALCON_STANDARD_SYNCLASS_OP_CREATE( Select, StmtSelect, zeroaryExprSet ) //
FALCON_STANDARD_SYNCLASS_OP_CREATE( Switch, StmtSwitch, zeroaryExprSet ) //
FALCON_STANDARD_SYNCLASS_OP_CREATE( Try, StmtTry, zeroaryExprSet ) //
FALCON_STANDARD_SYNCLASS_OP_CREATE( While, StmtWhile, zeroaryExprSet ) //

//=================================================================
// Statements
//
void SynClasses::ClassFastPrint::store( VMContext* ctx, DataWriter*wr, void* instance ) const
{
   StmtFastPrint* fp = static_cast<StmtFastPrint*>( instance );
   wr->write(fp->isAddNL());
   m_parent->store( ctx, wr, fp );
}

void SynClasses::ClassFastPrint::restore( VMContext* ctx, DataReader*dr ) const
{
   bool bHasNL;
   dr->read( bHasNL );
   StmtFastPrint* expr = new StmtFastPrint(bHasNL);

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

void SynClasses::ClassReturn::store( VMContext* ctx, DataWriter*wr, void* instance ) const
{
   StmtReturn* ret = static_cast<StmtReturn*>( instance );
   wr->write(ret->hasDoubt());
   wr->write(ret->hasEval());
   m_parent->store( ctx, wr, ret );
}

void SynClasses::ClassReturn::restore( VMContext* ctx, DataReader*dr ) const
{
   bool bHasDoubt;
   bool bHasEval;

   dr->read( bHasDoubt );
   dr->read( bHasEval );

   StmtReturn* expr = new StmtReturn;
   expr->hasDoubt(bHasDoubt);
   expr->hasEval(bHasEval);

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
      Symbol* tgt = forin->param(i);
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

         Symbol* sym = Engine::getSymbol(name);
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

void SynClasses::ClassSelect::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   StmtSelect* stmt = static_cast<StmtSelect*>(instance);
   TRACE1( "SynClasses::ClassSelect::flatten %s", stmt->oneLiner().c_ize() );
   stmt->flatten(ctx, subItems);
}
void SynClasses::ClassSelect::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   StmtSelect* stmt = static_cast<StmtSelect*>(instance);
   stmt->unflatten( ctx, subItems );
   TRACE1( "SynClasses::ClassForTo::unflatten %s", stmt->oneLiner().c_ize() );
}

void SynClasses::ClassForTo::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   static Class* clsSym = Engine::handlers()->symbolClass();

   StmtForTo* stmt = static_cast<StmtForTo*>(instance);
   TRACE1( "SynClasses::ClassForTo::flatten %s", stmt->oneLiner().c_ize() );
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
   TRACE1( "SynClasses::ClassForTo::unflatten %s", stmt->oneLiner().c_ize() );
}

//=================================================================
// Trees
//
FALCON_STANDARD_SYNCLASS_OP_CREATE( RuleSynTree, RuleSynTree, zeroaryExprSet )
   
}

/* end of synclasses.cpp */
