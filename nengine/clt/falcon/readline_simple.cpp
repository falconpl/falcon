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

using namespace Falcon;

void IntMode::read_line( const String& prompt, String &line )
{
   m_vm.textOut()->write(prompt);
   m_vm.textOut()->flush();
   
   uint32 maxSize = 4096; 
   line.reserve( maxSize );
   line.size(0);
   uint32 chr;
   while ( line.length() < maxSize && ( chr = m_vm->textIn()->getChar() ) != TextReader::NoChar )
   {
      if ( chr == '\r' )
         continue;
      if ( chr == '\n' )
         break;
      line += chr;
   }
}
