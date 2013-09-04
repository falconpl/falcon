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

   namespace {

static void internal_selector_add( VMContext* ctx, int32, bool bAdd )
{
   Item* i_stream = ctx->param(0);
   Item* i_mode = ctx->param(1);

   Class* cls = 0;
   void* data = 0;
   Selectable* resource = 0;
   if( i_stream == 0 || ! i_stream->asClassInst(cls,data)
       || ( resource = cls->getSelectableInterface(data) ) == 0
       || (i_mode != 0 && ! i_mode->isOrdinal() )
       )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC).extra( "resource:Selectable, mode:[N]") );
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
   if( bAdd )
   {
      sel->add( resource, mode );
   }
   else {
      sel->update( resource, mode );
   }

   ctx->returnFrame();
}



/*#
  @method add Selector
  @brief Adds a stream to the selector with a given mode
  @param resource The stream to be added
  @optparam mode Mode of read-write.

 */
FALCON_DECLARE_FUNCTION( add, "resource:<selectable>, mode:[N]" );

/*#
  @method update Selector
  @brief Changes the stream selection mode.
  @param resource The stream to be updated
  @param mode Mode of read-write.

 */
FALCON_DECLARE_FUNCTION( update, "resource:<selectable>, mode:N" );

/*#
  @method addRead Selector
  @brief Adds a stream to the selector for read operations
  @param resource The stream to be added for reading.
 */
FALCON_DECLARE_FUNCTION( addRead, "resource:<selectable>" );

/*#
  @method addWrite Selector
  @brief Adds a stream to the selector for write operations
  @param resource The stream to be added for writing.
 */
FALCON_DECLARE_FUNCTION( addWrite, "resource:<selectable>" );

/*#
  @method addWrite Selector
  @brief Adds a stream to the selector for errors and out-of-band operations
  @param resource The stream to be added for errors or out of band data.
 */
FALCON_DECLARE_FUNCTION( addErr, "resource:<selectable>" );



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
FALCON_DECLARE_FUNCTION( getRead, "" );

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
FALCON_DECLARE_FUNCTION( getWrite, "" );

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
FALCON_DECLARE_FUNCTION( getErr, "" );


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
FALCON_DECLARE_FUNCTION( get, "" );




void Function_add::invoke(VMContext* ctx, int32 pCount )
{
   internal_selector_add(ctx, pCount, true );
}


void Function_update::invoke(VMContext* ctx, int32 pCount )
{
   internal_selector_add(ctx, pCount, false );
}



static void internal_add_mode( VMContext* ctx, int32, int32 mode )
{
   Item* i_stream = ctx->param(0);

   Class* cls = 0;
   void* data = 0;
   Selectable* resource = 0;
   if( i_stream == 0 || ! i_stream->asClassInst(cls,data)
            || ( resource = cls->getSelectableInterface(data) ) == 0 )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC).extra( "resource:Selectable") );
   }

   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   sel->add( resource, mode );

   ctx->returnFrame();
}


void Function_addRead::invoke(VMContext* ctx, int32 pCount )

{
   internal_add_mode( ctx, pCount, Selector::mode_read );
}


void Function_addWrite::invoke(VMContext* ctx, int32 pCount )
{
   internal_add_mode( ctx, pCount, Selector::mode_write );
}


void Function_addErr::invoke(VMContext* ctx, int32 pCount )
{
   internal_add_mode( ctx, pCount, Selector::mode_err );
}


void Function_getRead::invoke(VMContext* ctx, int32 )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Selectable* resource = sel->getNextReadyRead();
   if( resource == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(resource->handler(), resource->instance()) );
   }
}


void Function_getWrite::invoke(VMContext* ctx, int32 )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Selectable* resource = sel->getNextReadyWrite();
   if( resource == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(resource->handler(), resource->instance()) );
   }
}


void Function_getErr::invoke(VMContext* ctx, int32 )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Selectable* resource = sel->getNextReadyErr();
   if( resource == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(resource->handler(), resource->instance()) );
   }
}


void Function_get::invoke(VMContext* ctx, int32 )
{
   Selector* sel = static_cast<Selector*>(ctx->self().asInst());
   Selectable* resource = sel->getNextReadyErr();
   if( resource == 0 ) resource = sel->getNextReadyRead();
   if( resource == 0 ) resource = sel->getNextReadyWrite();

   if( resource == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ctx->returnFrame( Item(resource->handler(), resource->instance()) );
   }
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

   addMethod( new Function_add);
   addMethod( new Function_update);
   addMethod( new Function_addRead);
   addMethod( new Function_addWrite);
   addMethod( new Function_addErr);

   addMethod( new Function_getRead);
   addMethod( new Function_getWrite);
   addMethod( new Function_getErr);
   addMethod( new Function_get);
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
