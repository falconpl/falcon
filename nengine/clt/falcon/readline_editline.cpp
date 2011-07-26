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
#include "editline/src/histedit.h"
#include <histedit.h>
#include <falcon/textreader.h>

using namespace Falcon;


bool IntMode::read_line( const String& prompt, String &line )
{
   const wchar_t *wline;
   static EditLine* el = 0;
   static HistoryW *hist = 0;
   static HistEventW ev;
   
   if( el == 0 )
   {
      el = el_init("falcon", stdin, stdout, stderr);
      hist = history_winit();
   }

   m_vm.textOut()->write(prompt);
   m_vm.textOut()->flush();
   
   int maxSize = 4096; 
   line.reserve( maxSize );
   line.size(0);   
      
   if( (wline = el_wgets(el, &maxSize)) != 0 )
   {
     if (maxSize > 0 )
     { 
        line += wline;
        //add_history(buf);
        history_w(hist, &ev, H_APPEND, wline);
        
     }
     return true;
   }
   else 
   {
      // EOF (CTRL-D)
      return false;
   }
}
