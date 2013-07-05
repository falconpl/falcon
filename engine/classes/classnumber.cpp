/*
 FALCON - The Falcon Programming Language.
 FILE: classnumber.cpp
 
 Function object handler.
 -------------------------------------------------------------------
 Author: Giancarlo Niccolai
 Begin: Sun, 10 Feb 2013 05:09:25 +0100
 
 -------------------------------------------------------------------
 (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)
 
 See LICENSE file for licensing details.
 */

#undef SRC
#define SRC "engine/classes/classnumber.cpp"


#include <falcon/classes/classnumber.h>
#include <falcon/itemid.h>
#include <falcon/item.h>
#include <falcon/optoken.h>
#include <falcon/vmcontext.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/stderrors.h>

#include <math.h>

namespace Falcon {
    

ClassNumber::ClassNumber() :
   Class( "Number", FLC_ITEM_NUM )
{ 
   m_bIsFlatInstance = true; 
}


ClassNumber::~ClassNumber()
{ 
}

void ClassNumber::dispose( void* ) const
{  
}


void *ClassNumber::clone( void* source ) const
{
   return source;
}

void *ClassNumber::createInstance() const
{
   return 0;
}

}

/* end of classnumber.cpp */

