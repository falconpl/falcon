/*
   FALCON - The Falcon Programming Language.
   FILE: sstream.cpp

   Falcon module interface for string streams.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 5 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @beginmodule core
*/

/** \file
   Falcon module interface for string streams.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/stringstream.h>


namespace Falcon {
namespace core {

/*#
   @class StringStream
   @brief Memory based stream.
   @optparam buffer The buffer that will be used as stream
   @from Stream

   The StringStream class inherits from stream. It can be used to provide
   functions that are supposed to write to streams with a memory
   buffer; for example, variables may be serialized on a string stream
   which can be then written completely on a physical stream, or sent over
   the network, or written in a database blob field. The reverse is of course
   possible: a string can be read from any source and then used to construct a
   StringStream, that can be then fed to function expecting streams as parameters.

   Of course, all the methods listed in the Stream class are available also here.

   The StringStream is always available for read and write operations, and
   supports seek operations. Writing past the end of the stream will cause
   the StringStream to grow.

   If the parameter @b buffer is a numeric value, the constructor preallocates
   the given size. Writes up to buffer size won't require re-allocation,
   and the size will be used as a hint to grow the stream buffer sensibly.

   If a string is provided, it is used as initial contents of the
   StringStream; subsequent reads will return the data contained in the string.
*/

FALCON_FUNC  StringStream_init ( ::Falcon::VMachine *vm )
{
   // check the paramenter.
   Item *size_itm = vm->param( 0 );
   Stream *stream;

   if ( size_itm != 0 )
   {
      if ( size_itm->isString() ) {
         stream = new StringStream ( *size_itm->asString() );
      }
      else if ( size_itm->isOrdinal() )
      {
         stream = new StringStream ( (int32) size_itm->forceInteger() );
      }
      else
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra( "S|N" ) );
      }
   }
   else
      stream = new StringStream ();

   // get the self object
   CoreObject *self = vm->self().asObject();

   // create the string stream
   self->setUserData( stream );
}

/*#
   @method getString StringStream
   @brief Returns the data currently stored in the stream.
   @return A copy of the contents of this stream.

   The data currently held in the stream is left untouched, and a new copy of the
   data is returned.
*/
FALCON_FUNC  StringStream_getString ( ::Falcon::VMachine *vm )
{
   // get the self object
   CoreObject *self = vm->self().asObject();
   StringStream *ss = (StringStream *)self->getUserData();
   vm->retval( ss->getCoreString() );
}

/*#
   @method closeToString StringStream
   @brief Close the stream and returns its contents.
   @return The stream contents.

   Closes the stream and returns the contents of the stream as a string.
   The object is internally destroyed, and the whole content is transformed
   into the returned string. In this way, an extra allocation and copy can be spared.
*/
FALCON_FUNC  StringStream_closeToString ( ::Falcon::VMachine *vm )
{
   // get the self object
   CoreObject *self = vm->self().asObject();
   StringStream *ss = (StringStream *)self->getUserData();
   CoreString *rets = new CoreString;
   ss->closeToString( *rets );
   vm->retval( rets );
}

}}
/* end of sstream.cpp */
