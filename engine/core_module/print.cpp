/*
   FALCON - The Falcon Programming Language.
   FILE: print.cpp

   Basic module
   -------------------------------------------------------------------
   Author: $AUTHOR
   Begin: $DATE

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/stream.h>

/*#
   
*/

namespace Falcon {
namespace core {

/*#
   @function print
   @inset core_basic_io
   @param ... An arbitrary list of parameters.
   @brief Prints the contents of various items to the standard output stream.

   This function is the default way for a script to say something to the outer
   world. Scripts can expect print to do a consistent thing with respect to the
   environment they work in; stand alone scripts will have the printed data
   to be represented on the VM output stream. The stream can be overloaded to
   provide application supported output; by default it just passes any write to
   the process output stream.

   The items passed to print are just printed one after another, with no separation.
   After print return, the standard output stream is flushed and the cursor (if present)
   is moved past the last character printed. The function @a printl must be used,
   or a newline character must be explicitly placed among the output items.

   The print function has no support for pretty print (i.e. numeric formatting, space
   padding and so on). Also, it does NOT automatically call the toString() method of objects.

   @see printl
*/
FALCON_FUNC  print ( ::Falcon::VMachine *vm )
{
   Stream *stream = vm->stdOut();
   if ( stream == 0 )
   {
      return;
   }

   for (int i = 0; i < vm->paramCount(); i ++ )
   {

      Item *elem = vm->param(i);
      String temp;

      switch( elem->type() ) {
         case FLC_ITEM_STRING:
            stream->writeString( *elem->asString() );
         break;

         default:
            elem->toString( temp );
            stream->writeString( temp );
      }
   }

   stream->flush();
}

/*#
   @function printl
   @inset core_basic_io
   @param ... An arbitrary list of parameters.
   @brief Prints the contents of various items to the VM standard output stream, and adds a newline.

   This functions works exactly as @a print, but it adds a textual "new line" after all the
   items are printed. The actual character sequence may vary depending on the underlying system.

   @see print
*/
FALCON_FUNC  printl ( ::Falcon::VMachine *vm )
{
   Stream *stream = vm->stdOut();
   if ( stream == 0 )
   {
      return;
   }

   print( vm );
   stream->writeString( "\n" );
   stream->flush();
}


}}

/* end of print.cpp */
