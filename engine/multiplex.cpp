/*
   FALCON - The Falcon Programming Language.
   FILE: multiplex.cpp

   Multiplex framework for Streams and Selectors.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Feb 2013 18:19:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/multiplex.cpp"

#include <falcon/multiplex.h>
#include <falcon/selector.h>
#include <falcon/module.h>


namespace Falcon
{

void Multiplex::gcMark( uint32 mark )
{
   if( m_mark != mark )
   {
      m_mark = mark;
      m_generator->gcMark(mark);
   }
}


void Multiplex::onReadyRead( Stream* stream )
{
   m_selector->pushReadyRead( stream );
}

void Multiplex::onReadyWrite( Stream* stream )
{
   m_selector->pushReadyWrite( stream );
}

void Multiplex::onReadyErr( Stream* stream )
{
   m_selector->pushReadyErr( stream );
}



void MultiplexGenerator::gcMark( uint32 mark )
{
   if( m_module != 0 && m_mark != mark )
   {
      m_mark = mark;
      m_module->gcMark(mark);
   }
}

}

/* end of multiplex.cpp */
