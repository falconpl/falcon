/*
   FALCON - The Falcon Programming Language
   FILE: regex_mod.cpp
   $Id: regex_mod.cpp,v 1.2 2006/11/21 15:54:00 gian Exp $

   Module support for Regular Expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 18 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Module support for Regular Expressions.
*/

#include "regex_mod.h"
#include <stdio.h>
#include <falcon/memory.h>

namespace Falcon {

RegexCarrier::RegexCarrier( pcre *pattern ):
   m_extra( 0 ),
   m_pattern( pattern )
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
