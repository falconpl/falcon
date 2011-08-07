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

#include <set>
#include <locale.h>

using namespace Falcon;

static wchar_t prompt_now[32];

static wchar_t *
prompt_func(EditLine *)
{
	return prompt_now;
}


static unsigned char
complete_func(EditLine *el, int)
{
   static std::set<String> m_stdFunctions;
   if ( m_stdFunctions.empty() )
   {
      m_stdFunctions.insert( "len" );
      m_stdFunctions.insert( "toString" );
      m_stdFunctions.insert( "clone" );
   }
   
	const wchar_t *ptr;
	const LineInfoW *lf = el_wline(el);
	unsigned char res = CC_NORM;

	/* Find the last word */
	for (ptr = lf->cursor -1; ptr > lf->buffer; --ptr)
   {
      if( String::isWhiteSpace(*ptr) || *ptr == '(' || *ptr == '.'  )
      {
         ++ptr;
         break;
      }
   }
   
	/* Scan directory for matching name */
   std::set<String>::iterator pos = m_stdFunctions.begin();
   while( pos != m_stdFunctions.end() )
   {
		if (pos->find( ptr ) == 0 ) 
      {
         el_deletestr( el, lf->cursor - ptr );
			if (el_insertstr(el, pos->c_ize()) == -1)
				res = CC_ERROR;
			else
				res = CC_REFRESH;
			break;
		}
      ++pos;
	}

	return res;
}


bool IntMode::read_line( const String& prompt, String &line )
{
   const wchar_t *wline;
   static EditLine* el = 0;
   static HistoryW *hist = 0;
   static HistEventW ev;
   
   // create the history.
   if( el == 0 )
   {
      setlocale(LC_ALL, "");
      el = el_init("falcon", stdin, stdout, stderr);      
      hist = history_winit();
      history_w(hist, &ev, H_SETSIZE, 5000);
      el_wset(el, EL_HIST, history_w, hist);
      
      el_wset(el, EL_EDITOR, L"emacs");
      el_wset(el, EL_SIGNAL, 1);		/* Handle signals gracefully */
      el_wset(el, EL_PROMPT_ESC, prompt_func, '\1'); /* Set the prompt function */

      /* Add a user-defined function	*/
      el_wset(el, EL_ADDFN, L"ed-complete", L"Complete argument", complete_func);

		/* Bind <tab> to it */
      el_wset(el, EL_BIND, L"^I", L"ed-complete", NULL);

      /* Source the user's defaults file. */
      el_source(el, NULL);
   }

   // set the prompt.
   prompt.toWideString( prompt_now, sizeof( prompt_now ) / sizeof(wchar_t) );   
      
   int maxSize = 4096; 
   line.reserve( maxSize );
   line.size(0);   
      
   if( (wline = el_wgets(el, &maxSize)) != 0 )
   {
     if (maxSize > 0 )
     { 
        line += wline;        
        history_w(hist, &ev, H_ENTER, wline);        
     }
     return true;
   }
   else 
   {
      // EOF (CTRL-D)
      return false;
   }
}
