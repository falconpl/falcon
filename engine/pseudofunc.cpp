/*
   FALCON - The Falcon Programming Language.
   FILE: pseudofunc.cpp

   Pseudo function definition and standard pseudo functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 07 May 2011 17:19:48 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/pseudofunc.h>

namespace Falcon {

PseudoFunction::PseudoFunction( const String& name, PStep* direct ):
   Function( name ),
   m_step(direct)
{
   m_category = e_c_pseudofunction;
}

PseudoFunction::~PseudoFunction()
{
}

}

/* end of pseudofunc.cpp */
