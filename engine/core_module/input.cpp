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
#include <errno.h>

#ifndef WIN32
   #include <unistd.h>
#else
   #include <io.h>
   #ifndef STDIN_FILENO
   #define STDIN_FILENO 0
   #endif
#endif

/*#
   
*/

namespace Falcon {
namespace core {

/*#
   @function input
   @inset core_basic_io
   @brief Get some text from the user (standard input stream).

   Reads a line from the standard input stream and returns a string
   containing the read data. This is mainly meant as a test/debugging
   function to provide the scripts with minimal console based user input
   support. When in need of reading lines from the standard input, prefer the
   readLine() method of the input stream object.

   This function may also be overloaded by embedders to provide the scripts
   with a common general purpose input function, that returns a string that
   the user is queried for.
*/
FALCON_FUNC  input ( ::Falcon::VMachine *vm )
{
   char mem[512];
   int size = 0;

   while( size < 511 )
   {
      if ( ::read( STDIN_FILENO, mem + size, 1 ) != 1 )
      {
            throw new IoError( ErrorParam( e_io_error )
            .origin( e_orig_runtime )
            .sysError( errno ) );
      }

      if( mem[size] == '\n' ) {
         mem[ size ] = 0;
         break;
      }
      else if( mem[size] != '\r' ) {
         size++;
      }
   }

   mem[size] = 0;
   CoreString *str = new CoreString;
   str->bufferize( mem );
   vm->retval( str );
}


}}
/* end of input.cpp */
