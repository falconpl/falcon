/*
   FALCON - The Falcon Programming Language.
   FILE: complex.cpp

   Complex class for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 09 Sep 2009 23:44:53 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.

*/

/** \file
   Complex class for Falcon
   Internal logic functions - implementation.
*/

#include <falcon/complex.h>
#include <falcon/error.h>
#include <falcon/eng_messages.h>

#include <math.h>

namespace Falcon {

void Complex::throw_div_by_zero()
{
   throw new MathError( ErrorParam( e_div_by_zero, __LINE__ )
      .origin( e_orig_runtime )
      .extra( "Complex number division by zero") );
}

numeric Complex::abs() const
{
   return sqrt( m_real * m_real + m_imag * m_imag );
}

Complex Complex::conj() const
{
   return Complex( m_real, m_imag * -1 );
}

}


/* end of complex.cpp */
