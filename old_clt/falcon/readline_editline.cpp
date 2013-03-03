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
#include <cstdio> // FILE
#ifdef FALCON_WITH_GPL_READLINE
# include <readline/readline.h>
# include <readline/history.h>
#else
# include <editline/readline.h>
#endif

using namespace Falcon;


namespace { // anonymous

Stream* s_in = 0;

extern "C" int s_getc(FILE*)
{
  uint32 chr;
  s_in->get(chr);
  return chr;
}

} // anonymous


void IntMode::read_line(String &line, const char* prompt)
{
//   if(!s_in)
//   {
//     s_in = m_owner->m_stdIn;
//     rl_getc_function = s_getc;
//   }
           
   line.reserve( 1024 );
   line.size(0);

   if( char* buf = readline(prompt) )
   { 
     if (buf[0] != 0)
     { 
       line += String(buf);
       add_history(buf);
     }
free(buf);
   }
   else // EOF (CTRL-D)
     m_owner->m_stdIn->status(Stream::t_eof);
}
