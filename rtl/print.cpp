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

namespace Falcon { namespace Ext {

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
