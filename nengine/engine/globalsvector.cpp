/*
   FALCON - The Falcon Programming Language.
   FILE: globalsvector.cpp

   Specialized vector holding global variables
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Feb 2011 11:10:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/globalsvector.h>
#include <falcon/globalsymbol.h>
#include <falcon/string.h>

namespace Falcon {

GlobalsVector::GlobalsVector( size_type size ):
   m_vector(size)
{
}

// here to prevent remote deletion
GlobalsVector::~GlobalsVector()
{
}


void GlobalsVector::resize( size_type size )
{
   m_vector.resize( size );
}


GlobalSymbol* GlobalsVector::makeGlobal( const String& name, size_type pos )
{
   if ( pos > m_vector.size() )
   {
      m_vector.resize( pos + 1 );
   }

   GlobalSymbol* sym = new GlobalSymbol( name, &m_vector[pos] );
   return sym;
}

}

/* end of globalsvector.cpp */
