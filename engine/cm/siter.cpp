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
#define SRC "engine/cm/siter.cpp"

#include <falcon/cm/siter.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/uri.h>
#include <falcon/path.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/function.h>

#include <falcon/stderrors.h>

/*#
 @beginmodule core
 */

namespace Falcon {
namespace Ext {

static void get_source( const Class*, const String&, void* instance, Item& value )
{
   SIterCarrier* ic = static_cast<SIterCarrier*>( instance );
   value = ic->source();
}


static void get_hasCurrent( const Class*, const String&, void* instance, Item& value )
{
   SIterCarrier* ic = static_cast<SIterCarrier*>( instance );
   value.setBoolean(ic->hasCurrent());
}

static void get_hasNext( const Class*, const String&, void* instance, Item& value )
{
   SIterCarrier* ic = static_cast<SIterCarrier*>( instance );
   value.setBoolean(ic->hasNext());
}

/*#
 @method reset SIter
 @brief moves the itrator back to the beginning of the sequence.
 @return self
 */

FALCON_DECLARE_FUNCTION( reset, "" );
void Function_reset::invoke(VMContext* ctx, int32 )
{
   SIterCarrier* ic = static_cast<SIterCarrier*>( ctx->self().asInst() );
   ClassSIter* cli = static_cast<ClassSIter*>(methodOf());
   cli->startIteration( ic, ctx );
   // do not return frame.
}


/*#
 @method current SIter
 @brief Gets the last element that was returned by next
 @raise AccessError if the iterator is invaid
 */
FALCON_DECLARE_FUNCTION( current, "" );
void Function_current::invoke(VMContext* ctx, int32 )
{
   SIterCarrier* ic = static_cast<SIterCarrier*>( ctx->self().asInst() );
   if( ! ic->hasCurrent() )
   {
      throw FALCON_SIGN_ERROR( AccessError, e_invalid_iter );

   }
   ctx->returnFrame(ic->current());
}

/*#
 @method next SIter
 @brief Gets the next element in the sequence.
 @raise AccessError if the iterator has not a next element.
 */

FALCON_DECLARE_FUNCTION( next, "" );
void Function_next::invoke(VMContext* ctx, int32 )
{
   SIterCarrier* ic = static_cast<SIterCarrier*>( ctx->self().asInst() );
   TRACE1( "ClassSimpleIterator::next %s", ctx->self().describe(2,50).c_ize() );

   if( ! ic->hasNext() )
   {
      throw FALCON_SIGN_ERROR( AccessError, e_invalid_iter );
   }

   if( ic->isFirst() )
   {
      ic->isFirst(false);
      ctx->returnFrame(ic->current());
      return;
   }

   Class* cls=0;
   void* inst=0;
   ic->source().asClassInst( cls, inst );

   ClassSIter* cli = static_cast<ClassSIter*>(methodOf());
   long depth = 0;
   ctx->pushCode( &cli->m_stepMethodNext_NextNext );
   depth = ctx->codeDepth();
   ctx->pushData( Item(cli, ic) );
   ctx->pushData( ic->source() );
   ctx->pushData( ic->intenralIter() );
   cls->op_next(ctx, inst);

   // descend immediately if possible
   if( ctx->codeDepth() == depth )
   {
      ctx->currentCode().m_step->apply( ctx->currentCode().m_step, ctx );
   }

   // otherwise, wait our turn to be called back
}


FALCON_DECLARE_FUNCTION( init, "sequence:[X]" );
void Function_init::invoke(VMContext* ctx, int32 pCount )
{
   SIterCarrier* ic = ctx->tself<SIterCarrier>();

   if ( pCount >= 1 )
   {
      Item& other = *ctx->param(0);
      ic->source().copyFromRemote(other);
      ClassSIter* cli = static_cast<ClassSIter*>(methodOf());
      cli->startIteration( ic, ctx );
      // do not return frame.
   }
   else
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_params,
            .extra( "X" )
            .origin(ErrorParam::e_orig_mod) );
   }

   // do not return frame -- startIteration shall return self.
}


//===================================================================================
//
//

/*#
 @class SIter
 @brief Allows to visit elements of classes providing a next operator.
 @param source The instance whose elements are to be visited.

 This class indirectly exposes the functionality of classes providing
 a "__iter()" and "__next()" methdos.

 Those methods are used by statements as the for/in, as well as pre-defined and core-module
 functions as classes, as @a advance(), @a map(), @a filter(), @a Generator and so on.

 A simple iterator instance can be used to programmatically pull one element
 at a time from the iterable sequence, as follows:

 @code
 seq = .[ 'a' 'b' 'c' 'd' 'e' ]
 i = SIter(seq)
 while i.hasNext
    > "Another element: ", i.next()
 end
 @endcode

 This class is not meant to provide container-depenedent abilities, as the ability
 to insert or remove an element. Hence the name SIter for "simple iterator". However, the
 method @a SIter.reset() can restore the iterator status to the initial value.

 @prop hasCurrent True if a current item can be accessed.
 @prop hasNext True if next() can be called to pull another element.
 @prop source The original sequence that is being iterated on.
*/

ClassSIter::ClassSIter():
   Class("SIter")
{
   setConstuctor( new Function_init );
   addProperty( "source", &get_source );
   addProperty( "hasCurrent", &get_hasCurrent );
   addProperty( "hasNext", &get_hasNext );
   addMethod( new Function_reset );
   addMethod( new Function_current );
   m_Method_next = new Function_next;
   addMethod( m_Method_next );
}


ClassSIter::~ClassSIter()
{}


void ClassSIter::startIteration( SIterCarrier* ic, VMContext* ctx )
{
   ic->isFirst(true);
   ic->hasCurrent(false);

   ctx->pushCode( &m_stepStartIterNext );
   ctx->pushData( Item(this, ic) );
   ctx->pushData( ic->source() );

   Class* cls=0;
   void* inst=0;
   ic->source().asClassInst( cls, inst );
   cls->op_iter(ctx, inst);
}


void ClassSIter::invokeDirectNextMethod( VMContext* ctx, void* instance, int32 pcount )
{
   ctx->self().setUser(this, instance);
   m_Method_next->invoke(ctx, pcount);
}


void ClassSIter::dispose( void* instance ) const
{
   SIterCarrier* ic = static_cast<SIterCarrier*>(instance);
   delete ic;
}

void* ClassSIter::clone( void* instance ) const
{
   SIterCarrier* ic = static_cast<SIterCarrier*>(instance);
   return ic->clone();
}

void ClassSIter::describe( void* instance, String& target, int depth, int maxlen ) const
{
   SIterCarrier* uc = static_cast<SIterCarrier*>(instance);
   if( maxlen > 0 )
   {
      maxlen =-12;
   }

   target = "Iterator(" + uc->source().describe(depth-1,maxlen) + ")";
}

void ClassSIter::gcMarkInstance( void* instance, uint32 mark ) const
{
   SIterCarrier* ic = static_cast<SIterCarrier*>(instance);
   ic->gcMark(mark);
}

bool ClassSIter::gcCheckInstance( void* instance, uint32 mark ) const
{
   SIterCarrier* ic = static_cast<SIterCarrier*>(instance);
   return ic->currentMark() >= mark;
}


void* ClassSIter::createInstance() const
{
   return new SIterCarrier;
}


void ClassSIter::store( VMContext*, DataWriter* stream, void* instance ) const
{
   SIterCarrier* ic = static_cast<SIterCarrier*>(instance);
   stream->write( ic->isFirst() );
   stream->write( ic->hasCurrent() );
}

void ClassSIter::restore( VMContext* ctx, DataReader* stream ) const
{
   bool bFirst = false;
   bool bReady = false;
   stream->read( bFirst );
   stream->read( bReady );

   SIterCarrier* ic = new SIterCarrier;
   ctx->pushData( Item(this, ic ) );
   ic->isFirst( bFirst );
   ic->hasCurrent( bReady );

}


void ClassSIter::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   SIterCarrier* ic = static_cast<SIterCarrier*>(instance);
   subItems.resize(3);
   subItems[0] = ic->source();
   subItems[1] = ic->intenralIter();
   subItems[2] = ic->current();
}


void ClassSIter::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   SIterCarrier* ic = static_cast<SIterCarrier*>(instance);
   fassert( subItems.length() == 3 );
   ic->source() = subItems[0];
   ic->intenralIter() = subItems[1];
   ic->current() = subItems[2];
}



// signature: (0: seq) -> (1: seq) (0: iter)
void ClassSIter::op_iter( VMContext* ctx, void* instance ) const
{
   SIterCarrier* ic = static_cast<SIterCarrier*>(instance);
   TRACE1( "ClassIterator::op_iter %s", Item(this, instance).describe(2,50).c_ize() );

   ctx->pushCode( &m_stepIterNext );

   CodeFrame& cc = ctx->currentCode();

   Class* cls=0;
   void* data=0;
   ic->source().asClassInst(cls, data);
   ctx->pushData( ic->source() );
   cls->op_iter( ctx, data );
   // did our iterator go deep?
   if( &cc != &ctx->currentCode() )
   {
      return;
   }

   // if not, we can invoke it now.
   m_stepIterNext.apply( &m_stepIterNext, ctx );
}

void ClassSIter::PStepIterNext::apply_( const PStep* , VMContext* ctx )
{
   TRACE1( "ClassIterator::PStepIterNext::apply %s", ctx->opcodeParam(2).describe(2,50).c_ize() );
   ctx->popCode();
   // signature: (2:ClassIterator) (1: seq) (0: iter) -> (1:ClassItertor) (1:Iter)

   SIterCarrier* ic = static_cast<SIterCarrier*>( ctx->opcodeParam(2).asInst() );
   ic->hasCurrent(true);
   ic->intenralIter() = ctx->opcodeParam(0);
   if( ic->intenralIter().isBreak() )
   {
      ic->current() = ic->intenralIter();
   }
   ctx->popData();
   ctx->topData() = ic->intenralIter();
}


/** Continues iteration.
 \b signature: (1: seq) (0: iter) --> (2: seq) (1: iter) (0: item|break)
 */
void ClassSIter::op_next( VMContext* ctx, void* instance ) const
{
   SIterCarrier* ic = static_cast<SIterCarrier*>(instance);
   TRACE1( "ClassIterator::op_next %s", Item(this, instance).describe(2,50).c_ize() );

   ctx->pushCode( &m_stepNextNext );

   CodeFrame& cc = ctx->currentCode();

   Class* cls=0;
   void* data=0;
   ic->source().asClassInst(cls, data);
   ctx->popData(); // remove the iterator, we have it.
   ctx->pushData( ic->source() );
   ctx->pushData( ic->intenralIter() );

   cls->op_next( ctx, data );
   // did our iterator go deep?
   if( &cc != &ctx->currentCode() )
   {
      return;
   }

   // if not, we can invoke it now.
   m_stepNextNext.apply( &m_stepNextNext, ctx );
}


void ClassSIter::PStepNextNext::apply_( const PStep*, VMContext* ctx )
{
   TRACE1( "ClassIterator::PStepNextNext::apply %s", ctx->opcodeParam(2).describe(2,50).c_ize() );

   //(3:ClassIterator) (2: seq) (1: iter) (0: item|break) -> (2:ClassIterator) (1: iter) (0: item|break)
   ctx->popCode();

   SIterCarrier* ic = static_cast<SIterCarrier*>( ctx->opcodeParam(3).asInst() );
   ic->hasCurrent(true);
   ic->intenralIter() = ctx->opcodeParam(1);
   Item temp = ctx->opcodeParam(0);
   ctx->popData(3);
   ctx->pushData(ic->intenralIter());
   ctx->pushData(temp);
   ic->current() = temp;

   if( temp.isBreak() )
   {
      ic->hasCurrent(false);
      ic->intenralIter().setNil();
   }
}


void ClassSIter::PStepStartIterNext::apply_( const PStep* , VMContext* ctx )
{
   TRACE1( "ClassIterator::PStepStartIterNext::apply %s", ctx->opcodeParam(2).describe(2,50).c_ize() );
   // signature: (2:ClassIterator) (1: seq) (0: iter) -> (2:ClassIterator) (1: seq) (0: iter)  -- same

   SIterCarrier* ic = static_cast<SIterCarrier*>( ctx->opcodeParam(2).asInst() );
   ic->intenralIter() = ctx->opcodeParam(0);
   if( ic->intenralIter().isBreak() || ic->intenralIter().isNil() )
   {
      throw FALCON_SIGN_ERROR( AccessError, e_invalid_iter );
   }

   // get the first element
   Class* cls = 0;
   void* data = 0;
   ctx->opcodeParam(1).asClassInst(cls, data);
   ClassSIter* cli = static_cast<ClassSIter*>(ctx->opcodeParam(2).asClass());
   ctx->pushCode( &cli->m_stepStartNextNext );
   cls->op_next(ctx,data);

   // no need to pop the code, we're going to return the frame
}


void ClassSIter::PStepStartNextNext::apply_( const PStep* , VMContext* ctx )
{
   TRACE1( "ClassIterator::PStepStartNextNext::apply %s", ctx->opcodeParam(2).describe(2,50).c_ize() );
   // signature: (3:ClassIterator) (2:seq) (1:iter) (0:value) -> return self

   // no need to pop the code, we're going to return the frame

   SIterCarrier* ic = static_cast<SIterCarrier*>( ctx->opcodeParam(3).asInst() );
   ic->current() = ctx->opcodeParam(0);
   ic->hasCurrent(!ic->current().isBreak());
   // we always need to copy our iterator, because some classes have flat iterators.
   ic->intenralIter() = ctx->opcodeParam(1);
   // return self.
   ctx->returnFrame(ctx->self());
}

void ClassSIter::PStepMethodNext_NextNext::apply_( const PStep* , VMContext* ctx )
{
   TRACE1( "ClassIterator::PStepMethodNext_NextNext::apply %s", ctx->opcodeParam(3).describe(2,50).c_ize() );
   // signature: (3:ClassIterator) (2: seq) (1: iter) (0: item|break) -> return frame
   Item ret = ctx->opcodeParam(0);
   SIterCarrier* ic = static_cast<SIterCarrier*>( ctx->opcodeParam(3).asInst() );
   // we always need to copy our iterator, because some classes have flat iterators.
   ic->intenralIter() = ctx->opcodeParam(1);
   ic->current() = ret;

   if( ret.isBreak() )
   {
      ic->hasCurrent(false);
      // be sure to do a clean job.
      if( ctx->paramCount() != 0 )
      {
         ret = *ctx->param(0);
      }
      else {
         ret.type( FLC_ITEM_NIL );
      }
      ret.setBreak();
   }

   // returnframe will keep the doubt flag of the item.
   ctx->returnFrame( ret );
}

}
}

/* end of iterator.cpp */
