/*
   FALCON - The Falcon Programming Language.
   FILE: metafalconclass.cpp

   Handler for classes defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 10 Mar 2012 23:27:16 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classes/metafalconclass.h>

namespace Falcon
{

MetaFalconClass::MetaFalconClass()
{
   m_name = "Class";
}

MetaFalconClass::~MetaFalconClass()
{
}


void MetaFalconClass::store( VMContext* , DataWriter* , void*  ) const
{
   
}

void MetaFalconClass::restore( VMContext* , DataReader* , void*&  ) const
{
   
}

}

/* end of metafalconclass.cpp */
