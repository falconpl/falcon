/*
   FALCON - The Falcon Programming Language
   FILE: regex_mod.cpp

   Module support for Regular Expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 18 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Module support for Regular Expressions.
*/

#include "regex_mod.h"
#include <stdio.h>
#include <falcon/memory.h>

namespace Falcon {

RegexCarrier::RegexCarrier( pcre *pattern ):
   m_pattern( pattern ),
   m_extra( 0 ),
   m_matches(0)
{
   int retval;
   pcre_fullinfo( pattern, 0, PCRE_INFO_CAPTURECOUNT, &retval );
   m_ovectorSize = (retval + 2) * 3;
   m_ovector = (int *) memAlloc( m_ovectorSize * sizeof( int ) );
}


RegexCarrier::~RegexCarrier()
{
   memFree( m_ovector );

   pcre_free( m_pattern );

   if( m_extra != 0 )
   {
      if ( m_extra->study_data != 0 )
         pcre_free( m_extra->study_data );
      pcre_free( m_extra );
   }
}

}

/* end of regex_mod.cpp */
