/*
   FALCON - The Falcon Programming Language.
   FILE: sourceref.h

   Reference to a point in a source file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:39:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_SOURCEREF_H
#define FALCON_SOURCEREF_H

#include <falcon/setup.h>
#include <falcon/item.h>

namespace Falcon
{

class Stream;

class FALCON_DYN_CLASS SourceRef
{
public:
   inline SourceRef( int line=0, int chr=0 ):
      m_line( line ),
      m_char( chr )
   {
   }

   inline SourceRef( const SourceRef& other ):
      m_line( other.m_line ),
      m_char( other.m_char )
   {}

   inline int32 line() const { return m_line; }
   inline int32 chr() const { return m_char; }

   void serialize( Stream* s ) const;
   void deserialize( Stream* s );

private:
   int32 m_line;
   int32 m_char;
};

}

#endif

/* end of sourceref.h */

