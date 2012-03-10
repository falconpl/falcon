/*
   FALCON - The Falcon Programming Language.
   FILE: metahyperclass.cpp

   Handler for classes defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 10 Mar 2012 23:27:16 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classes/metahyperclass.h>

namespace Falcon
{

MetaHyperClass::MetaHyperClass()
{
   m_name = "$HyperClass";
}

MetaHyperClass::~MetaHyperClass()
{
}


void MetaHyperClass::store( VMContext* , DataWriter* , void*  ) const
{
   
}

void MetaHyperClass::restore( VMContext* , DataReader* , void*&  ) const
{
   
}

}

/* end of metahyperclass.cpp */
