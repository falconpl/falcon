/*
   FALCON - The Falcon Programming Language.
   FILE: requiredclass.cpp

   Structure holding information about inheritance in a class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:58:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/requiredclass.cpp"

#include <falcon/requirement.h>
#include <falcon/class.h>

namespace Falcon
{

Requirement::Requirement( const String& name, Requirer* req ):
   m_name( name ),
   m_req( req )
{}
   
Requirement::~Requirement()
{
}

Error* Requirement::resolve( Module* mod, const Symbol* sym )
{
   return m_req->resolved( mod, sym, this );
}


}

/* end of requiredclass.cpp */
