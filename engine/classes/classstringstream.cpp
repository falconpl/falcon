/*
   FALCON - The Falcon Programming Language.
   FILE: classstringstream.cpp

   Falcon core module -- String stream interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/stringstream.cpp"

#include <falcon/classes/classstream.h>
#include <falcon/classes/classstringstream.h>
#include <falcon/errors/paramerror.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/stringstream.h>
#include <falcon/stdhandlers.h>

#include <falcon/vm.h>

namespace Falcon {

/*#
 @property pipeMode StringStream
 @brief Gets or sets the pipe mode for this string stream.

 In pipe mode, the string stream read and write pointers are different.

 If the mode is set to false, they move together, and the write pointer
 is reset to the position of the read pointer.

 In pipe mode, seek() moves both the read and the write pointer,
 and current position is relative to write pointer,
 but tell() returns the read pointer.
*/
void get_pipeMode( const Class*, const String&, void *instance, Item& value )
{
   StringStream* ss = static_cast<StringStream*>(instance);
   value.setBoolean(ss->isPipeMode());
}

void set_pipeMode( const Class*, const String&, void *instance, const Item& value )
{
   StringStream* ss = static_cast<StringStream*>(instance);
   ss->setPipeMode(value.isTrue());
}

/*#
 @property content StringStream

 @brief The whole content of this string stream as a memory buffer string.
*/
void get_content( const Class*, const String&, void *instance, Item& value )
{
   StringStream* ss = static_cast<StringStream*>(instance);
   String* ret = new String;
   ss->getString(*ret);
   ret->manipulator(ret->manipulator()->membufManipulator());
   value = FALCON_GC_HANDLE( ret );
}


/*#
 @method closeToString StringStream

 @brief Gets the data written to the stream in a memory-efficient way.

 Closes the string and passes the string memory as-is to a memory buffer string.
 After this call the stream is not usable anymore.
*/
FALCON_DECLARE_FUNCTION(closeToString, "");
void Function_closeToString::invoke(Falcon::VMContext *ctx, Falcon::int32)
{
   StringStream* ss = static_cast<StringStream*>(ctx->self().asInst());

   String* ret = ss->closeToString();
   ret->manipulator(ret->manipulator()->membufManipulator());
   ctx->returnFrame( FALCON_GC_HANDLE( ret ) );
}



ClassStringStream::ClassStringStream():
      ClassStream("StringStream")
{
   static Class* ssc = Engine::handlers()->streamClass();
   setParent(ssc);
   addProperty( "pipeMode", &get_pipeMode, &set_pipeMode );
   addProperty( "content", &get_content );
   addMethod( new Function_closeToString );
}

ClassStringStream::~ClassStringStream()
{}

void* ClassStringStream::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassStringStream::op_init( VMContext* ctx, void*, int pcount ) const
{
   int64 count;

   if( pcount == 0 )
   {
      count = 0;
   }
   else {
      Item* i_count = ctx->opcodeParams(pcount);
      if( ! i_count->isOrdinal() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("N") );
      }
      count = i_count->forceInteger();
      if( count < 0 )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra(">=0") );
      }
   }

   StringStream* ss = new StringStream( (int32) count );
   ctx->stackResult(pcount+1, FALCON_GC_STORE(this, ss));
   return true;
}

}

/* end of classstringstream.cpp */
