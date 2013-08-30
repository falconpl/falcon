/*
   FALCON - The Falcon Programming Language.
   FILE: selector.cpp

   Interface for script to Shared variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Nov 2012 12:52:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classselector.cpp"

#include <falcon/selector.h>
#include <falcon/cm/selector.h>
#include <falcon/stream.h>
#include <falcon/stderrors.h>
#include <falcon/shared.h>
#include <falcon/vm.h>

#include <falcon/vmcontext.h>
#include <falcon/stdhandlers.h>
#include <falcon/stdsteps.h>

namespace Falcon {
namespace Ext {


static void internal_selector_add( VMContext* ctx, int32, bool bAdd )
{
   static Class* streamClass = Engine::handlers()->streamClass();
   Item* i_stream = ctx->param(0);
   Item* i_mode = ctx->param(1);

   Class* cls = 0;
   void* data = 0;
   if( i_stream == 0 || ! i_stream->asClassInst(cls,data) || ! cls->isDerivedFrom(streamClass)
       || (i_mode != 0 && ! i_mode->isOrdinal() )
       )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC).extra( "stream:Stream, mode:[N]") );
   }

   int32 mode;
   if( i_mode == 0 )
   {
      mode = Selector::mode_err | Selector::mode_write | Selector::mode_read;
   }
   else
   {
      mode = (int32)i_mode->forceInteger();
      if( mode < 0 || mode > (Selector::mode_err | Selector::mode_write | Selector::mode_read) )
      {
         throw new ParamError(ErrorParam(e_param_range, __LINE__, SRC).extra( "select mode out of range") );
      }
   }

   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* sc = static_cast<Stream*>( cls->getParentData(streamClass,data) );
   if( bAdd )
   {
      sel->add( sc, mode );
   }
   else {
      sel->update( sc, mode );
   }

   ctx->returnFrame();
}



/*#
  @method add Selector
  @brief Adds a stream to the selector with a given mode
  @param stream The stream to be added
  @optparam mode Mode of read-write.

 */
FALCON_DECLARE_FUNCTION( selector_add, "stream:Stream, mode:[N]" );

/*#
  @method update Selector
  @brief Changes the stream selection mode.
  @param stream The stream to be updated
  @param mode Mode of read-write.

 */
FALCON_DECLARE_FUNCTION( selector_update, "stream:Stream, mode:N" );

/*#
  @method addRead Selector
  @brief Adds a stream to the selector for read operations
  @param stream The stream to be added for reading.
 */
FALCON_DECLARE_FUNCTION( selector_addRead, "stream:Stream" );

/*#
  @method addWrite Selector
  @brief Adds a stream to the selector for write operations
  @param stream The stream to be added for writing.
 */
FALCON_DECLARE_FUNCTION( selector_addWrite, "stream:Stream" );

/*#
  @method addWrite Selector
  @brief Adds a stream to the selector for errors and out-of-band operations
  @param stream The stream to be added for errors or out of band data.
 */
FALCON_DECLARE_FUNCTION( selector_addErr, "stream:Stream" );



/*#
  @method getRead Selector
  @brief Gets the next ready-for-read stream.
  @return Next ready stream.

  If this selector is created in fair mode, the get operation will
  exit the critical section acquired at wait success.

  If the method is not in fair mode, and there are multiple waiters
  for this selector, the get method might return nil, as the ready
  streams might get dequeued by other agents before this method is
  complete.

  Also, the method will return nil if the wait successful for other
  operations.
 */
FALCON_DECLARE_FUNCTION( selector_getRead, "" );

/*#
  @method getWrite Selector
  @brief Gets the next ready-for-write stream.
  @return Next ready stream.

  If this selector is created in fair mode, the get operation will
  exit the critical section acquired at wait success.

  If the method is not in fair mode, and there are multiple waiters
  for this selector, the get method might return nil, as the ready
  streams might get dequeued by other agents before this method is
  complete.

  Also, the method will return nil if the wait successful for other
  operations.
 */
FALCON_DECLARE_FUNCTION( selector_getWrite, "" );

/*#
  @method getErr Selector
  @brief Gets the next out-of-band signaled stream.
  @return Next ready stream.

  If this selector is created in fair mode, the get operation will
  exit the critical section acquired at wait success.

  If the method is not in fair mode, and there are multiple waiters
  for this selector, the get method might return nil, as the ready
  streams might get dequeued by other agents before this method is
  complete.

  Also, the method will return nil if the wait successful for other
  operations.
 */
FALCON_DECLARE_FUNCTION( selector_getErr, "" );


/*#
  @method get Selector
  @brief Gets the next ready stream.
  @return Next ready stream.

  This method returns the first ready stream, peeked in order in:
  - Out of band data signaled queue.
  - Ready for read signaled queue.
  - Ready for write signaled queue.

  The method doesn't indicate what the operation the stream is ready
  for; so if this method is used, other means need to be available
  to know how to use the returned stream.

  If this selector is created in fair mode, the get operation will
  exit the critical section acquired at wait success.

  If the method is not in fair mode, and there are multiple waiters
  for this selector, the get method might return nil, as the ready
  streams might get dequeued by other agents before this method is
  complete.

  Also, the method will return nil if the wait successful for other
  operations.
 */
FALCON_DECLARE_FUNCTION( selector_get, "" );




void Function_selector_add::invoke(VMContext* ctx, int32 pCount )
{
   internal_selector_add(ctx, pCount, true );
}


void Function_selector_update::invoke(VMContext* ctx, int32 pCount )
{
   internal_selector_add(ctx, pCount, false );
}



static void internal_add_mode( VMContext* ctx, int32, int32 mode )
{
   static Class* streamClass = Engine::handlers()->streamClass();
   Item* i_stream = ctx->param(0);

   Class* cls = 0;
   void* data = 0;
   if( i_stream == 0 || ! i_stream->asClassInst(cls,data) || ! cls->isDerivedFrom(streamClass) )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC).extra( "stream:Stream") );
   }

   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* sc = static_cast<Stream*>( cls->getParentData(streamClass,data) );
   sel->add( sc, mode );

   ctx->returnFrame();
}


void Function_selector_addRead::invoke(VMContext* ctx, int32 pCount )

{
   internal_add_mode( ctx, pCount, Selector::mode_read );
}


void Function_selector_addWrite::invoke(VMContext* ctx, int32 pCount )
{
   internal_add_mode( ctx, pCount, Selector::mode_write );
}


void Function_selector_addErr::invoke(VMContext* ctx, int32 pCount )
{
   internal_add_mode( ctx, pCount, Selector::mode_err );
}


void Function_selector_getRead::invoke(VMContext* ctx, int32 )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* stream = sel->getNextReadyRead();
   if( stream == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(stream->handler(), stream) );
   }
}


void Function_selector_getWrite::invoke(VMContext* ctx, int32 )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* stream = sel->getNextReadyWrite();
   if( stream == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(stream->handler(), stream) );
   }
}


void Function_selector_getErr::invoke(VMContext* ctx, int32 )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* stream = sel->getNextReadyErr();
   if( stream == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(stream->handler(), stream) );
   }
}


void Function_selector_get::invoke(VMContext* ctx, int32 )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Stream* stream = sel->getNextReadyErr();
   if( stream == 0 ) stream = sel->getNextReadyRead();
   if( stream == 0 ) stream = sel->getNextReadyWrite();

   if( stream == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(stream->handler(), stream) );
   }
}



//=========================================================
//
//=========================================================


ClassSelector::ClassSelector():
         ClassShared("Selector")
{
   static Class* shared = Engine::handlers()->sharedClass();
   setParent( shared );

   addMethod( new Function_selector_add);
   addMethod( new Function_selector_update);
   addMethod( new Function_selector_addRead);
   addMethod( new Function_selector_addWrite);
   addMethod( new Function_selector_addErr);

   addMethod( new Function_selector_getRead);
   addMethod( new Function_selector_getWrite);
   addMethod( new Function_selector_getErr);
   addMethod( new Function_selector_get);
}


ClassSelector::~ClassSelector()
{
}


void* ClassSelector::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassSelector::op_init( VMContext* ctx, void*, int pcount ) const
{
   Item* i_mode = ctx->opcodeParams(pcount);

   bool fair = i_mode == 0 ? false : i_mode->isTrue();
   Selector* sel = new Selector( &ctx->vm()->contextManager(), this, fair );
   ctx->stackResult(pcount+1, FALCON_GC_HANDLE(sel) );

   return true;
}

}
}

/* end of selector.cpp */
