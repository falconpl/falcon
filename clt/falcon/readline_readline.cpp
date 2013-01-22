/*
   FALCON - The Falcon Programming Language.
   FILE: readline_editline.cpp

   Falcon compiler and interpreter - interactive mode
   -------------------------------------------------------------------
   Author: Maik Beckmann
   Begin: Mon, 30 Oct 2009

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "int_mode.h"

using namespace Falcon;

# include <readline/readline.h>
# include <readline/history.h>

bool IntMode::read_line( const String& prompt, String &line )
{      
   uint32 maxSize = 4096; 
   line.reserve( maxSize );
   line.size(0);   
   char* buf;
   
   if( (buf = readline(prompt.c_ize())) != 0 )
   {
     if (buf[0] != 0)
     { 
        line += buf;
        add_history(buf);
     }
free(buf);
     return true;
   }
   else 
   {
      // EOF (CTRL-D)
      return false;
   }
}
