/*
   FALCON - The Falcon Programming Language.
   FILE: input.cpp

   Read a line from the input stream
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom set 19 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Read a line from the input stream.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/memory.h>

#include <stdio.h>

#ifndef WIN32
   #include <unistd.h>
#else
   #include <io.h>
   #ifndef STDIN_FILENO
   #define STDIN_FILENO 0
   #endif
#endif

namespace Falcon { namespace Ext {

FALCON_FUNC  input ( ::Falcon::VMachine *vm )
{
   char mem[512];
   int size = 0;

   while( size < 511 )
   {
      ::read( STDIN_FILENO, mem + size, 1 );

      if( mem[size] == '\n' ) {
         mem[ size ] = 0;
         break;
      }
      else if( mem[size] != '\r' ) {
         size++;
      }
   }

   mem[size] = 0;
   String *str = new GarbageString( vm );
   str->bufferize( mem );
   vm->retval( str );
}


}}
/* end of input.cpp */
