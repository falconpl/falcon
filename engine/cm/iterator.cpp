/*
   FALCON - The Falcon Programming Language.
   FILE: iterator.cpp

   Falcon core module -- Iterator
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Feb 2013 21:20:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/iterator.cpp"

#include <falcon/cm/iterator.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/uri.h>
#include <falcon/path.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>

#include <falcon/errors/paramerror.h>
#include <falcon/errors/accesserror.h>

namespace Falcon {
namespace Ext {



ClassIterator::ClassIterator():
   ClassUser("Iterator"),
   FALCON_INIT_METHOD( next ),
   FALCON_INIT_METHOD( rewind ),
   FALCON_INIT_PROPERTY( source )
{
}

ClassIterator::~ClassIterator()
{}


void ClassIterator::invokeDirectNextMethod( VMContext* ctx, void* instance, int32 pcount )
{
   ctx->self().setUser(this, instance);
   m_Method_next.invoke(ctx, pcount);
}


void ClassIterator::dispose( void* instance ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance);
   delete ic;
}

void* ClassIterator::clone( void* instance ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance);
   return ic->clone();
}

void ClassIterator::describe( void* instance, String& target, int depth, int maxlen ) const
{
   IteratorCarrier* uc = static_cast<IteratorCarrier*>(instance);
   if( maxlen > 0 )
   {
      maxlen =-12;
   }

   target = "Iterator(" + uc->m_source.describe(depth-1,maxlen) + ")";
}

void ClassIterator::gcMarkInstance( void* instance, uint32 mark ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance);
   if( ic->m_mark != mark )
   {
      ic->m_mark = mark;
      ic->m_source.gcMark(mark);
      ic->m_srciter.gcMark(mark);
   }
}

bool ClassIterator::gcCheckInstance( void* instance, uint32 mark ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance);
   return ic->m_mark >= mark;
}


void* ClassIterator::createInstance() const
{
   return new IteratorCarrier;
}


bool ClassIterator::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance );
   
   if ( pcount >= 1 )
   {
      Item& other = *ctx->opcodeParams(pcount);
      ic->m_source.assignFromRemote(other);
   }
   else
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_params,
            .extra( "X" )
            .origin(ErrorParam::e_orig_mod) );
   }
   
   return false;
}


void ClassIterator::store( VMContext*, DataWriter* stream, void* instance ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance);
   stream->write( ic->m_ready );
}

void ClassIterator::restore( VMContext* ctx, DataReader* stream ) const
{
   bool bReady = false;
   stream->read( bReady );

   IteratorCarrier* ic = new IteratorCarrier;
   ctx->pushData( Item(this, ic ) );
   ic->m_ready = bReady;
}


void ClassIterator::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance);
   subItems.resize(2);
   subItems[0] = ic->m_source;
   subItems[1] = ic->m_srciter;
}


void ClassIterator::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance);
   fassert( subItems.length() == 2 );
   ic->m_source = subItems[0];
   ic->m_srciter = subItems[1];
}



// signature: (0: seq) -> (1: seq) (0: iter)
void ClassIterator::op_iter( VMContext* ctx, void* instance ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance);
   TRACE1( "ClassIterator::op_iter %s", Item(this, instance).describe(2,50).c_ize() );

   ctx->pushCode( &m_stepIterNext );

   CodeFrame& cc = ctx->currentCode();

   Class* cls;
   void* data;
   ic->m_source.asClassInst(cls, data);
   ctx->pushData( ic->m_source );
   cls->op_iter( ctx, data );
   // did our iterator go deep?
   if( &cc != &ctx->currentCode() )
   {
      return;
   }

   // if not, we can invoke it now.
   m_stepIterNext.apply( &m_stepIterNext, ctx );
}

void ClassIterator::PStepIterNext::apply_( const PStep* , VMContext* ctx )
{
   TRACE1( "ClassIterator::PStepIterNext::apply %s", ctx->opcodeParam(2).describe(2,50).c_ize() );
   ctx->popCode();
   // signature: (2:ClassIterator) (1: seq) (0: iter) -> (1:ClassItertor) (1:Iter)

   IteratorCarrier* ic = static_cast<IteratorCarrier*>( ctx->opcodeParam(2).asInst() );
   ic->m_ready = true;
   ic->m_srciter = ctx->opcodeParam(0);
   ctx->popData();
   ctx->topData() = ic->m_srciter;
}


/** Continues iteration.
 \b signature: (1: seq) (0: iter) --> (2: seq) (1: iter) (0: item|break)
 */
void ClassIterator::op_next( VMContext* ctx, void* instance ) const
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>(instance);
   TRACE1( "ClassIterator::op_next %s", Item(this, instance).describe(2,50).c_ize() );

   ctx->pushCode( &m_stepNextNext );

   CodeFrame& cc = ctx->currentCode();

   Class* cls;
   void* data;
   ic->m_source.asClassInst(cls, data);
   ctx->popData(); // remove the iterator, we have it.
   ctx->pushData( ic->m_source );
   ctx->pushData( ic->m_srciter );

   cls->op_next( ctx, data );
   // did our iterator go deep?
   if( &cc != &ctx->currentCode() )
   {
      return;
   }

   // if not, we can invoke it now.
   m_stepNextNext.apply( &m_stepNextNext, ctx );
}


void ClassIterator::PStepNextNext::apply_( const PStep*, VMContext* ctx )
{
   TRACE1( "ClassIterator::PStepNextNext::apply %s", ctx->opcodeParam(2).describe(2,50).c_ize() );

   //(3:ClassIterator) (2: seq) (1: iter) (0: item|break) -> (2:ClassIterator) (1: iter) (0: item|break)
   ctx->popCode();

   IteratorCarrier* ic = static_cast<IteratorCarrier*>( ctx->opcodeParam(3).asInst() );
   ic->m_ready = true;
   ic->m_srciter = ctx->opcodeParam(1);
   Item temp = ctx->opcodeParam(0);
   ctx->popData(3);
   ctx->pushData(ic->m_srciter);
   ctx->pushData(temp);

   if( temp.isBreak() )
   {
      ic->m_ready = false;
      ic->m_srciter.setNil();
   }
}


FALCON_DEFINE_PROPERTY_GET_P( ClassIterator, source )
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>( instance );
   value = ic->m_source;
}

FALCON_DEFINE_PROPERTY_SET_P0( ClassIterator, source )
{
   throw FALCON_SIGN_ROPROP_ERROR( "source" );
}


FALCON_DEFINE_METHOD_P1( ClassIterator, rewind )
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>( ctx->self().asInst() );
   ic->m_ready = false;
   ic->m_srciter.setNil();
   ctx->returnFrame();
}

FALCON_DEFINE_METHOD_P1( ClassIterator, next )
{
   IteratorCarrier* ic = static_cast<IteratorCarrier*>( ctx->self().asInst() );
   TRACE1( "ClassIterator::next %s", ctx->self().describe(2,50).c_ize() );

   Class* cls;
   void* inst;
   ic->m_source.asClassInst( cls, inst );

   ClassIterator* cli = static_cast<ClassIterator*>(methodOf());
   long depth = 0;
   if( ! ic->m_ready )
   {
      ctx->pushCode( &cli->m_stepMethodNext_NextNext );
      ctx->pushCode( &cli->m_stepMethodNext_IterNext );
      depth = ctx->codeDepth();
      ctx->pushData( Item(cli, ic) );
      ctx->pushData( ic->m_source );
      cls->op_iter(ctx, inst);
   }
   else {
      ctx->pushCode( &cli->m_stepMethodNext_NextNext );
      depth = ctx->codeDepth();
      ctx->pushData( Item(cli, ic) );
      ctx->pushData( ic->m_source );
      ctx->pushData( ic->m_srciter );
      cls->op_next(ctx, inst);
   }

   // descend immediately if possible
   if( ctx->codeDepth() == depth )
   {
      ctx->currentCode().m_step->apply( ctx->currentCode().m_step, ctx );
   }

   // otherwise, wait our turn to be called back
}

void ClassIterator::PStepMethodNext_IterNext::apply_( const PStep* , VMContext* ctx )
{
   TRACE1( "ClassIterator::PStepMethodNext_IterNext::apply %s", ctx->opcodeParam(2).describe(2,50).c_ize() );
   // signature: (2:ClassIterator) (1: seq) (0: iter) -> (3:ClassIterator) (2: seq) (1: iter) (0: item|break)
   ctx->popCode();

   IteratorCarrier* ic = static_cast<IteratorCarrier*>( ctx->opcodeParam(2).asInst() );
   ic->m_ready = true;
   ic->m_srciter = ctx->opcodeParam(0);

   Class* cls;
   void* inst;
   ic->m_source.asClassInst( cls, inst );

   long depth = ctx->codeDepth();
   cls->op_next( ctx, inst );

   // descended?
   if( depth == ctx->codeDepth() )
   {
      // no ? -- enter the next step now.
      ctx->currentCode().m_step->apply( ctx->currentCode().m_step, ctx );
      return;
   }

   // else wait to be called back;
}

void ClassIterator::PStepMethodNext_NextNext::apply_( const PStep* , VMContext* ctx )
{
   TRACE1( "ClassIterator::PStepMethodNext_NextNext::apply %s", ctx->opcodeParam(3).describe(2,50).c_ize() );
   // signature: (3:ClassIterator) (2: seq) (1: iter) (0: item|break) -> return frame
   Item ret = ctx->opcodeParam(0);
   IteratorCarrier* ic = static_cast<IteratorCarrier*>( ctx->opcodeParam(3).asInst() );
   // we always need to copy our iterator, because some classes have flat iterators.
   ic->m_srciter = ctx->opcodeParam(1);

   if( ret.isBreak() )
   {
      // be sure to do a clean job.
      if( ctx->paramCount() != 0 )
      {
         ret = *ctx->param(0);
         ret.setBreak();
      }
      else {
         ret.type( FLC_ITEM_NIL );
      }

   }

   // returnframe will keep the doubt flag of the item.
   ctx->returnFrame( ret );
}

}
}

/* end of iterator.cpp */
