/*
   FALCON - The Falcon Programming Language.
   FILE: mxml_mod.cpp

   Minimal XML processor module - module specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Mar 2008 14:44:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Minimal XML processor module - module specific extensions.
*/

#include "mxml_mod.h"

namespace Falcon {
namespace Ext {

//===================================================================
// Node carrier
//

NodeCarrier::~NodeCarrier()
{
   // if the carried node has a parent, just unreserve it,
   // else, we have to destroy it.
   if ( m_node->parent() != 0 || m_node->isReserved() )
   {
      m_node->shell( 0 );
   }
   else
      delete m_node;
}

UserData *NodeCarrier::clone() const
{
   MXML::Node *node = m_node->clone();
   return new NodeCarrier( node, 0 );
}


}
}

/* end of mxml_mod.cpp */
