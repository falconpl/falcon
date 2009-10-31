/*
   FALCON - The Falcon Programming Language.
   FILE: readline_simple.cpp

   Falcon compiler and interpreter - interactive mode
   -------------------------------------------------------------------
   Author: Maik Beckmann
   Begin: Mon, 30 Oct 2009

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "int_mode.h"

void IntMode::read_line(String &line, const char* prompt)
{
   m_owner->m_stdOut->writeString(prompt);
   m_owner->m_stdOut->flush();
   uint32 maxSize = 1024; 
   line.reserve( maxSize );
   line.size(0);
   uint32 chr;
   while ( line.length() < maxSize && m_owner->m_stdIn->get( chr ) )
   {
      if ( chr == '\r' )
         continue;
      if ( chr == '\n' )
         break;
      line += chr;
   }
}
