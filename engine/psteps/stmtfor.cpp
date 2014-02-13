/*
   FALCON - The Falcon Programming Language.
   FILE: stmtfor.cpp

   Syntactic tree item definitions -- Autoexpression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Aug 2011 17:28:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtfor.cpp"

#include <falcon/trace.h>
#include <falcon/fassert.h>
#include <falcon/expression.h>
#include <falcon/stdsteps.h>
#include <falcon/syntree.h>
#include <falcon/symbol.h>
#include <falcon/itemarray.h>
#include <falcon/textwriter.h>

#include <falcon/stderrors.h>

#include <falcon/psteps/stmtfor.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprvalue.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>

#include <vector>

namespace Falcon 
{


StmtForBase::~StmtForBase()
{
   dispose( m_body );
   dispose( m_forFirst );
   dispose( m_forMiddle );
   dispose( m_forLast );
}

void StmtForBase::minimize()
{
   m_body = minimize_basic( m_body );
   m_forFirst = minimize_basic( m_forFirst );
   m_forMiddle = minimize_basic( m_forMiddle );
   m_forLast = minimize_basic( m_forLast );
}

void StmtForBase::body( TreeStep* st ) {
   if( st->setParent(this) )
   {
      dispose( m_body );
      m_body = st;
   }
}

void StmtForBase::forFirst( TreeStep* st )
{
   if( st->setParent(this) )
   {
      dispose( m_forFirst );
      m_forFirst = st;
   }
}

void StmtForBase::forMiddle( TreeStep* st ) {
   if( st->setParent(this) )
   {
      dispose( m_forMiddle );
      m_forMiddle = st;
   }
}

void StmtForBase::forLast( TreeStep* st ) {
   if( st->setParent(this) )
   {
      dispose( m_forLast );
      m_forLast = st;
   }
}


int32 StmtForBase::arity() const
{
   return 4;
}

TreeStep* StmtForBase::nth( int32 n ) const
{
   switch( n )
   {
      case 0: case -4: return m_body;
      case 1: case -3: return m_forFirst;
      case 2: case -2: return m_forMiddle;
      case 3: case -1: return m_forLast;      
   }
   return 0;
}

bool StmtForBase::setNth( int32 n, TreeStep* ts )
{
   // accept even a 0
   if( ts != 0 )
   {
      if( ts->parent() != 0 || n < -4 || n > 3 ) return false;

      if( ts->category() != TreeStep::e_cat_syntree ) {
         SynTree* st = new SynTree(ts->sr().line(), ts->sr().chr());
         st->append(ts);
         ts = st;
      }
   }
     
   switch( n )
   {
      case 0: case -4: dispose( m_body ); m_body = static_cast<SynTree*>(ts); break;
      case 1: case -3: dispose( m_forFirst ); m_forFirst = static_cast<SynTree*>(ts); break;
      case 2: case -2: dispose( m_forMiddle ); m_forMiddle = static_cast<SynTree*>(ts); break;
      case 3: case -1: dispose( m_forLast ); m_forLast = static_cast<SynTree*>(ts); break;
      default: return false;
   }
   
   if( ts != 0 )
   {
      ts->setParent( this );
   }

   return true;
}

   
void StmtForBase::PStepCleanup::apply_( const PStep*, VMContext* ctx )
{
   TRACE( "StmtForBase::PStepCleanup::apply_ %d (-1)",  ctx->currentCode().m_seqId );
   ctx->restoreUnrollPoint();
   ctx->popCode();
   // Add the value for the loop
   ctx->pushData(Item());
}


StmtForBase::StmtForBase( const StmtForBase& other ):
   Statement( other ),
   m_body(0),
   m_forFirst(0),
   m_forMiddle(0),
   m_forLast(0)
{
   if( other.m_body != 0) 
   {
      m_body = other.m_body->clone(); 
      m_body->setParent(this);
   }

   if( other.m_forFirst != 0) 
   {
      m_forFirst = other.m_forFirst->clone(); 
      m_forFirst->setParent(this);
   }

   if( other.m_forMiddle != 0) 
   {
      m_forMiddle = other.m_forMiddle->clone(); 
      m_forMiddle->setParent(this);
   }

   if( other.m_forLast != 0) 
   {
      m_forLast = other.m_forLast->clone(); 
      m_forLast->setParent(this);
   }   
}

void StmtForBase::render( TextWriter* tw, int32 depth ) const
{
   if( ! isValid() )
   {
      tw->write( renderPrefix(depth) );
      tw->write( "/* Blank StmtForIn/to */" );
   }
   else
   {
      tw->write( renderPrefix(depth) );
      renderHeading( tw, depth );
      tw->write("\n");

      int dp = depth < 0 ? -depth : depth;
      if( m_body != 0 )
      {
         m_body->render( tw, dp+1);
      }

      if( m_forFirst != 0 )
      {
         tw->write( renderPrefix(dp) );
         tw->write("forfirst\n");
         m_forFirst->render( tw, dp + 1 );
         tw->write( renderPrefix(dp) );
         tw->write("end\n");
      }

      if( m_forMiddle != 0 )
      {
         tw->write( renderPrefix(dp) );
         tw->write("formiddle\n");
         m_forMiddle->render( tw, dp + 1 );
         tw->write( renderPrefix(dp) );
         tw->write("end\n");
      }

      if( m_forLast != 0 )
      {
         tw->write( renderPrefix(dp) );
         tw->write("forlast\n");
         m_forLast->render( tw, dp + 1 );
         tw->write( renderPrefix(dp) );
         tw->write("end\n");
      }

      tw->write( renderPrefix(depth) );
      tw->write("end");
   }

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}

bool StmtForBase::setBodyFromParam(Item* param)
{
   if( param == 0 || param->isNil() ) {
      dispose(m_body);
      m_body = 0;
      return true;
   }
   else if( param->type() == FLC_CLASS_ID_TREESTEP ){
      TreeStep* ts = static_cast<TreeStep*>(param->asInst() );
      if( ts->setParent( this ) ) {
         dispose(m_body);
         m_body = ts;
         return true;
      }
   }

   return false;
}

bool StmtForBase::setForFirstFromParam(Item* param)
{
   if( param == 0 || param->isNil() ) {
      dispose(m_forFirst);
      m_forFirst = 0;
      return true;
   }
   else if( param->type() == FLC_CLASS_ID_TREESTEP ){
      TreeStep* ts = static_cast<TreeStep*>(param->asInst() );
      if( ts->setParent( this ) ) {
         dispose(m_forFirst);
         m_forFirst = ts;
         return true;
      }
   }

   return false;
}

bool StmtForBase::setForMiddleFromParam(Item* param)
{
   if( param == 0 || param->isNil() ) {
      dispose(m_forMiddle);
      m_forMiddle = 0;
      return true;
   }
   else if( param->type() == FLC_CLASS_ID_TREESTEP ){
      TreeStep* ts = static_cast<TreeStep*>(param->asInst() );
      if( ts->setParent( this ) ) {
         dispose(m_forMiddle);
         m_forMiddle = ts;
         return true;
      }
   }

   return false;
}

bool StmtForBase::setForLastFromParam(Item* param)
{
   if( param == 0 || param->isNil() ) {
      dispose(m_forLast);
      m_forLast = 0;
      return true;
   }
   else if( param->type() == FLC_CLASS_ID_TREESTEP ){
      TreeStep* ts = static_cast<TreeStep*>(param->asInst() );
      if( ts->setParent( this ) ) {
         dispose(m_forLast);
         m_forLast = ts;
         return true;
      }
   }

   return false;
}


//=================================================================
// For - in
//

class StmtForIn::Private
{
public:
   typedef std::vector<const Symbol*> SymVector;
   SymVector m_params;
   
   Private() {}
   
   Private( const Private& other):
      m_params( other.m_params )
   {}
   
   ~Private() {
      clearParams();
   }

   void clearParams()
   {
      SymVector::iterator iter = m_params.begin();
      while( iter != m_params.end() )
      {
         const Symbol* sym = *iter;
         sym->decref();
         ++iter;
      }
   }
};

StmtForIn::StmtForIn( int32 line, int32 chr):
   StmtForBase( line, chr ),
   _p( new Private ),
   m_expr(0),
   m_stepBegin( this ),
   m_stepFirst( this ),
   m_stepGetFirst( this ),
   m_stepNext( this ),
   m_stepGetNext( this )
{
   FALCON_DECLARE_SYN_CLASS(stmt_forin)
   apply = apply_; 
   
   //NOTE: This pstep is NOT a loopbase; it just sets up the loop.
   // a break here must be intercepted by outer loops until our loop is setup.
}

StmtForIn::StmtForIn( Expression* gen, int32 line, int32 chr):
   StmtForBase( line, chr ),
   _p( new Private ),
   m_expr(gen),
   m_stepBegin( this ),
   m_stepFirst( this ),
   m_stepGetFirst( this ),
   m_stepNext( this ),
   m_stepGetNext( this )
{
   FALCON_DECLARE_SYN_CLASS(stmt_forin)
   apply = apply_; 
   gen->setParent(this);
   
   //NOTE: This pstep is NOT a loopbase; it just sets up the loop.
   // a break here must be intercepted by outer loops until our loop is setup.
}

StmtForIn::StmtForIn( const StmtForIn& other ):
   StmtForBase( other ),
   _p( new Private(*other._p) ),
   m_expr(0),
   m_stepBegin( this ),
   m_stepFirst( this ),
   m_stepGetFirst( this ),
   m_stepNext( this ),
   m_stepGetNext( this )
{
   apply = apply_;
   
   if( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);
   }
}

StmtForIn::~StmtForIn()
{
   delete _p;
}

bool StmtForIn::isValid() const 
{
   return m_expr != 0 && _p->m_params.size() != 0 ;
}

void StmtForIn::renderHeading( TextWriter* tw, int32 depth ) const
{
   tw->write("for ");
   String syms;
   Private::SymVector::const_iterator iter = _p->m_params.begin();
   while( iter != _p->m_params.end() )
   {
      if( iter != _p->m_params.begin() )
      {
         syms += ", ";
      }
      tw->write((*iter)->name());
      ++iter;
   }

   tw->write(" in ");
   fassert( m_expr != 0 );
   m_expr->render(tw, relativeDepth(depth));
}


void StmtForIn::addParameter( const Symbol* sym )
{
   _p->m_params.push_back( sym );
}

TreeStep*  StmtForIn::selector() const
{
   return generator();
}

bool StmtForIn::selector( TreeStep* e )
{
   if( e != 0 && e->setParent( this ) )
   {
      dispose( m_expr );
      m_expr = e;
      return true;
   }
   return false;
}


length_t StmtForIn::paramCount() const
{
   return (length_t) _p->m_params.size();
}

const Symbol* StmtForIn::param( length_t p ) const
{
   return _p->m_params[p];
}


bool StmtForIn::setTargetFromParam(Item* param)
{
   if( param->type() == FLC_CLASS_ID_SYMBOL )
   {
      const Symbol* sym = static_cast<Symbol*>(param->asInst());
      _p->clearParams();
      _p->m_params.push_back(sym);
      sym->incref();
      return true;
   }
   else if( param->isArray() )
   {
      ItemArray* array = param->asArray();
      _p->clearParams();
      for( length_t i = 0; i < array->length(); ++i )
      {
         Item& value = array->at(i);
         if( value.type() == FLC_CLASS_ID_SYMBOL )
         {
            const Symbol* sym = static_cast<Symbol*>(value.asInst());
            sym->incref();
            _p->m_params.push_back(sym);
         }
         else {
            return false;
         }
      }

      return true;
   }

   return false;
}


bool StmtForIn::setSelectorFromParam(Item* param)
{
   if( param == 0 )
   {
      return false;
   }

   bool bCreated = true;
   Expression* result = checkExpr(*param, bCreated );
   if( result != 0 )
   {
      selector(result);
      return true;
   }

   return false;
}


void StmtForIn::expandItem( Item& itm, VMContext* ctx ) const
{
   if( _p->m_params.size() == 1 )
   {
      ctx->resolveSymbol(_p->m_params[0], true)->copyInterlocked(itm);
   }
   else
   {
      register Item* dr = &itm;   
      if( dr->isArray() )
      {
         ItemArray* ar = dr->asArray();
         if( ar->length() == _p->m_params.size() )
         {
            for( length_t i = 0; i < ar->length(); ++i )
            {
               ctx->resolveSymbol(_p->m_params[i], true)->copyInterlocked((*ar)[i]);
            }
            return;
         }
         
      }
      
      // failed...
      throw new AccessError( ErrorParam( e_unpack_size, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm)
         .extra( "for/in") 
         );
   }
}

void StmtForIn::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForIn* self = static_cast<const StmtForIn*>(ps);
   
   fassert( self->isValid() );
   
   // prepare the unroll point for the whole loop
   ctx->currentCode().m_step = &self->m_stepCleanup;
   ctx->saveUnrollPoint( ctx->currentCode() );

   ctx->pushCode( &self->m_stepBegin );
   ctx->stepIn( self->m_expr );
}

void StmtForIn::PStepBegin::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForIn* self = static_cast<const StmtForIn::PStepBegin*>(ps)->m_owner;
   
   // we have the evaluated expression on top of the stack -- make it to next.
   Class* cls;
   void* dt;
   if( ctx->topData().asClassInst( cls, dt ) )
   {       
      // Prepare to get the first op_next after op_iter
      ctx->resetCode( &self->m_stepGetFirst );
      
      // and create an iterator.
      cls->op_iter( ctx, dt );
   }
   else if( ctx->topData().isNil() )
   {
      // Nil is defined to cleanly NOT TO ENTER the loop.
      // keep the nil as for/in result
      ctx->popCode();
   }
   else if( ctx->topData().isMethod() )
   {
      ctx->topData().forceClassInst(cls, dt);
      ctx->resetCode( &self->m_stepGetFirst );
      cls->op_iter( ctx, dt );
   }
   else
   {
      throw new CodeError( 
         ErrorParam( e_not_iterable, __LINE__, SRC )
         .origin( ErrorParam::e_orig_vm )
         .extra( "for/in" ) );
   }    
}

void StmtForIn::PStepGetFirst::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForIn::PStepGetFirst* step = static_cast<const StmtForIn::PStepGetFirst*>(ps);
   fassert( ctx->opcodeParam(1).isUser() || ctx->opcodeParam(1).isMethod() );

   // Nil is the non-iterator
   if( ctx->topData().isNil() )
   {
      ctx->popCode();
      ctx->popData();
      ctx->topData().setNil();
      return;
   }

   // we're never needed anymore
   ctx->resetCode(&step->m_owner->m_stepFirst);

   Class* cls = 0;
   void* dt = 0;
   // here we have seq, iter, <space>...
   ctx->opcodeParam(1).forceClassInst( cls, dt );
   // ... pass them to next.
   cls->op_next( ctx, dt );
}


void StmtForIn::PStepGetNext::apply_( const PStep*, VMContext* ctx )
{
   ctx->restoreUnrollPoint();

   fassert( ctx->opcodeParam(1).isUser() || ctx->opcodeParam(1).isMethod() );

   // we're never needed anymore -- stepNext is behind us
   ctx->popCode();

   Class* cls = 0;
   void* dt = 0;
   // here we have seq, iter, <space>...
   ctx->opcodeParam(1).forceClassInst( cls, dt );
   // ... pass them to next.
   cls->op_next( ctx, dt );
}


void StmtForIn::PStepFirst::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForIn* self = static_cast<const StmtForIn::PStepFirst*>(ps)->m_owner;

   // we have here seq, iter, item
   Item& topData = ctx->topData();
   if( topData.isBreak() )
   {
      // the loop is over
      ctx->popCode();
      return;
   }

   // prepare the loop variabiles.
   self->expandItem( topData, ctx );
   
   if( ! topData.isDoubt() )
   {
      ctx->popData();
      ctx->popCode(); // we won't be called again.
      if( self->m_forLast != 0 )
      {
         ctx->pushCode( self->m_forLast );
      }
   }
   else
   {
      ctx->popData();
      // turn this step in the next step
      ctx->currentCode().m_step = &self->m_stepNext;
      // and save the get next with next landing
      ctx->pushCodeWithUnrollPoint( &self->m_stepGetNext );
      
      if ( self->m_forMiddle != 0 )
      {
         ctx->pushCode( self->m_forMiddle );
      }
   }
   
   if ( self->m_body != 0 )
   {
      ctx->pushCode( self->m_body );
   }

   if ( self->m_forFirst != 0 )
   {
      ctx->pushCode( self->m_forFirst );
   }
}


void StmtForIn::PStepNext::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForIn* self = static_cast<const StmtForIn::PStepNext*>(ps)->m_owner;

   // we have here seq, iter, item
   Item& topData = ctx->topData();
   if( topData.isBreak() )
   {
      ctx->popCode();
      return;
   }
   
   // prepare the loop variabiles.
   self->expandItem( ctx->topData(), ctx );

   if( ! topData.isDoubt() )
   {
      ctx->popData();
      ctx->popCode(); // we won't be called again.
       
      if( self->m_forLast != 0 )
      {
         ctx->pushCode( self->m_forLast );
      }
   }
   else
   {
      ctx->popData();
      // save the next step with landing
      ctx->pushCodeWithUnrollPoint( &self->m_stepGetNext );
      if ( self->m_forMiddle != 0 )
      {
         ctx->pushCode( self->m_forMiddle );
      }
   }
      
   if ( self->m_body != 0 )
   {
      ctx->pushCode( self->m_body );
   }

}


//===============================================
// For - to
//

StmtForTo::StmtForTo( const Symbol* tgt, Expression* start, Expression* end, Expression* step, int32 line, int32 chr ):
   StmtForBase( line, chr ),
   m_target( tgt ),
   m_start(start),
   m_end(end),  
   m_step(step),
   m_stepNext(this)
{
   FALCON_DECLARE_SYN_CLASS(stmt_forto)   
   apply = apply_;

   // act as a loop-base in case for/to expression wants to break.
   m_bIsLoopBase = true;
}


StmtForTo::StmtForTo( const StmtForTo& other ):
   StmtForBase( other ),
   m_target( other.m_target ),
   m_start(0),
   m_end(0),  
   m_step(0),
   m_stepNext(this)
{
   apply = apply_;
   m_target->incref();
   
   // act as a loop-base in case for/to expression wants to break.
   m_bIsLoopBase = true;

   if( other.m_start != 0 ) {
      m_start = other.m_start->clone();
      m_start->setParent(this);
   }
   
   if( other.m_end != 0 ) {
      m_end = other.m_end->clone();
      m_end->setParent(this);
   }
   
   if( other.m_step != 0 ) {
      m_step = other.m_step->clone();
      m_step->setParent(this);
   }
   
}

StmtForTo::~StmtForTo() 
{
   if( m_target != 0 )
   {
      m_target->decref();
   }
   dispose( m_start );
   dispose( m_end );
   dispose( m_step );
}

bool StmtForTo::isValid() const 
{
   return m_target != 0 && m_start != 0 && m_end != 0;
}

void StmtForTo::startExpr( Expression* s )
{
   if( s->setParent(this) )
   {
      dispose( m_start );
      m_start = s;
   }
}



bool StmtForTo::setTargetFromParam(Item* param)
{
   if ( param == 0 )
   {
      return false;
   }

   if( param->type() == FLC_CLASS_ID_SYMBOL )
   {
      const Symbol* sym = static_cast<Symbol*>(param->asInst());

      sym->incref();
      if( m_target != 0 )
      {
         m_target->decref();
      }
      m_target = sym;
      return true;
   }

   return false;
}


bool StmtForTo::setSelectorFromParam(Item* )
{
   return false;
}


bool StmtForTo::setStartExprFromParam(Item* param)
{
   if ( param == 0 )
   {
      return false;
   }

   if( param->type() == FLC_CLASS_ID_SYMBOL )
   {
      const Symbol* sym = static_cast<Symbol*>(param->asInst());
      startExpr(new ExprSymbol(sym ));
      return true;
   }
   else if( param->isOrdinal() )
   {
      startExpr(new ExprValue(*param));
      return true;
   }
   else if( param->type() == FLC_CLASS_ID_TREESTEP )
   {
      Expression* expr = static_cast<Expression*>( param->asInst() );
      if( expr->category() == TreeStep::e_cat_expression && expr->parent() == 0 )
      {
         startExpr( expr );
         return true;
      }
   }

   return false;
}


bool StmtForTo::setEndExprFromParam(Item* param)
{
   if ( param == 0 )
   {
      return false;
   }

   if( param->type() == FLC_CLASS_ID_SYMBOL )
   {
      const Symbol* sym = static_cast<Symbol*>(param->asInst());
      endExpr(new ExprSymbol(sym ));
      return true;
   }
   else if( param->isOrdinal() )
   {
      endExpr(new ExprValue(*param));
      return true;
   }
   else if( param->type() == FLC_CLASS_ID_TREESTEP )
   {
      Expression* expr = static_cast<Expression*>( param->asInst() );
      if( expr->category() == TreeStep::e_cat_expression && expr->parent() == 0 )
      {
         endExpr( expr );
         return true;
      }
   }

   return false;
}


bool StmtForTo::setStepExprFromParam(Item* param)
{
   if ( param == 0 || param->isNil() )
   {
      // step is optional
      dispose( m_step );
      m_step = 0;
      return true;
   }

   if( param->type() == FLC_CLASS_ID_SYMBOL )
   {
      const Symbol* sym = static_cast<Symbol*>(param->asInst());
      stepExpr(new ExprSymbol(sym ));
      return true;
   }
   else if( param->isOrdinal() )
   {
      stepExpr(new ExprValue(*param));
      return true;
   }
   else if( param->type() == FLC_CLASS_ID_TREESTEP )
   {
      Expression* expr = static_cast<Expression*>( param->asInst() );
      if( expr->category() == TreeStep::e_cat_expression && expr->parent() == 0 )
      {
         stepExpr( expr );
         return true;
      }
   }

   return false;
}


void StmtForTo::endExpr( Expression* s )
{
   if( s->setParent(this) )
   {
      dispose( m_end );
      m_end = s;
   }
}
   
void StmtForTo::stepExpr( Expression* s )
{
   if( s->setParent(this) )
   {
      dispose( m_step );
      m_step = s;
   }
}


void StmtForTo::renderHeading( TextWriter* tw, int32 depth ) const
{
   tw->write("for ");
   tw->write( m_target->name() );
   tw->write( " = " );
   
   m_start->render(tw, relativeDepth(depth));
   tw->write( " to ");
   m_end->render(tw, relativeDepth(depth));
   if( m_step != 0 )
   {
      tw->write(", ");
      m_step->render(tw, relativeDepth(depth));
   }
}


void StmtForTo::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtForTo* self = static_cast<const StmtForTo*>(ps);
   
   TRACE( "StmtForTo::PStepNext::apply_ %d/3",  ctx->currentCode().m_seqId );

   fassert( self->isValid() );

   // we must at least have a start and an end
   fassert( self->m_start != 0 );
   fassert( self->m_end != 0 );
   
   // First of all, start executing the start, end and step expressions.
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      case 0: 
         cf.m_seqId = 1;
         // save the points.
         ctx->saveUnrollPoint(cf);

         // check the start.
         if( ctx->stepInYield( self->m_start, cf ) )
         {
            return;
         }
         /* no break */

      case 1:
         cf.m_seqId = 2;
         if( ctx->stepInYield( self->m_end, cf ) )
         {
            return;
         }
         /* no break */

      case 2:
         cf.m_seqId = 3;
         if( self->m_step != 0 )
         {
            if( ctx->stepInYield( self->m_step, cf ) )
            {
               return;
            }
         }
         else {
            ctx->pushData( (int64) 0 );
         }
         break;
   }
   
   if( ! ctx->opcodeParam(0).isOrdinal()
            || ! ctx->opcodeParam(1).isOrdinal()
            || ! ctx->opcodeParam(2).isOrdinal() )
   {
      throw new TypeError( ErrorParam( e_for_not_numeric, __LINE__, SRC ) );
   }

   if( ! ctx->topData().isInteger() ) ctx->topData().setInteger(ctx->topData().forceInteger());
   if( ! ctx->opcodeParam(1).isInteger() ) ctx->opcodeParam(1).setInteger(ctx->opcodeParam(1).forceInteger());
   if( ! ctx->opcodeParam(2).isInteger() ) ctx->opcodeParam(2).setInteger(ctx->opcodeParam(1).forceInteger());

   int64 step = ctx->topData().asInteger();
   int64 end = ctx->opcodeParam(1).asInteger();
   int64 start = ctx->opcodeParam(2).asInteger();

   
   // in some cases, we don't even start the loop
   if( (end > start && step < 0) || (start > end && step > 0 ) )
   {
      ctx->popCode();
      ctx->popData(2);
      ctx->topData().setNil();
      return;
   }
   
   // fix the default step
   if( step == 0 )
   {
      if ( end >= start ) 
      {
         ctx->opcodeParam(0).setInteger(1);
      }
      else
      {
         ctx->opcodeParam(0).setInteger(-1);
      }      
   }
   
   // however, we won't be called anymore (cleanup is also the loop base)
   cf.m_step = &self->m_stepCleanup;
   // I have already saved the unroll points at seqID 0

   ctx->pushCodeWithUnrollPoint( &self->m_stepNext );
   
   // Prepare the start value   
   const Symbol* target = self->m_target;
   ctx->resolveSymbol(target, true)->copyFromLocal(Item(start));
   
   // eventually, push the first opode in top of all.
   if( self->m_forFirst != 0 )
   {
      ctx->pushCode( self->m_forFirst );
   }
}
 

void StmtForTo::PStepNext::apply_( const PStep* ps, VMContext* ctx )
{
   MESSAGE( "StmtForTo::PStepNext::apply_" );
   const StmtForTo* self = static_cast<const StmtForTo::PStepNext*>(ps)->m_owner;
   
   // reset the unroll point for next.
   ctx->restoreUnrollPoint();

   register int64 start = ctx->opcodeParam(2).asInteger();
   int64 end = ctx->opcodeParam(1).asInteger();
   int64 step = ctx->topData().asInteger();
   
   // the start, at minimum, will be done.
   const Symbol* target = self->m_target;
   ctx->resolveSymbol(target, true)->setInteger(start);
   start += step;

   // step cannot be 0 as it has been sanitized by our main step.
   if( (step > 0 && start > end) || ( step < 0 && start < end ) )
   {
      // this will be the last loop.
      ctx->popCode();
      
      if( self->m_forLast != 0 )
      {
         ctx->pushCode( self->m_forLast );
      }
   }
   else
   {
      if( self->m_forMiddle != 0 )
      {
         ctx->pushCode( self->m_forMiddle );
      }
   }
   
   if( self->m_body )
   {
      ctx->pushCode( self->m_body );
   }
   
   ctx->opcodeParam(2).content.data.val64 = start;
}


}

/* end of stmtfor.cpp */
